import sys
import os
import time
import shutil
import random
import warnings
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import sklearn.utils.validation
from sklearn.base import BaseEstimator, RegressorMixin

# --- 1. PATches & SETTINGS ---

# 1.1 Warnungen unterdrücken
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NumPy array is not writable*", category=UserWarning)

# 1.2 Sklearn Patch für TabPFN (zwingend für neuere Scikit-Learn Versionen)
if not hasattr(sklearn.utils.validation, "_is_pandas_df"):
    def _is_pandas_df(X):
        return hasattr(X, "dtypes") and hasattr(X, "columns")


    sklearn.utils.validation._is_pandas_df = _is_pandas_df

# Erst JETZT TabPFN importieren
from tabpfn import TabPFNRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent if "__file__" in locals() else Path(
    ".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.hpo.hpo_pipeline.data_loader import DataLoader
from src.eval.evaluation.loader import load_test_data_v2

# --- 2. CONFIGURATION ---

CONFIG = {
    "HPO_SAMPLE_N": 100000,  # <--- Reduziert auf 10k (V2)
    "TARGET_VARIABLE": "label_5min",
    "CHUNK_SIZE": 256,  # Wir können größere Chunks nehmen, da Inference schneller ist
    "OUTPUT_DIR": PROJECT_ROOT / "data/parquet/predictions/distributed/tabpfn_reg_100k_77f"
}


# --- 3. ROBUST MODEL WRAPPER ---

class TabPFNRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=1, max_train_samples=100000, device='cpu'):
        self.n_estimators = n_estimators
        self.max_train_samples = max_train_samples
        self.device = device
        self.model = None

    def _clean_input(self, X):
        """
        Kritische Funktion: Macht Daten TabPFN-kompatibel (float32, writable, contiguous).
        """
        # Falls DataFrame: Metadaten droppen, falls noch vorhanden
        if isinstance(X, pd.DataFrame):
            cols_to_drop = ['delivery_start', 'snapshot_times']
            X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
            X = X.select_dtypes(include=[np.number, bool]).values

        # Explizite Kopie im RAM erzeugen -> macht es 'writable' und 'contiguous'
        X_clean = np.array(X, copy=True).astype(np.float32)
        X_clean.setflags(write=True)
        return X_clean

    def fit(self, X, y):
        if self.model is None:
            self.model = TabPFNRegressor(
                device=self.device,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=True
            )

        X_clean = self._clean_input(X)
        y_np = y.values if hasattr(y, "values") else y
        y_np = np.array(y_np, copy=True).astype(np.float32)

        # Sampling falls Context zu groß
        if len(X_clean) > self.max_train_samples:
            idx = np.random.choice(len(X_clean), self.max_train_samples, replace=False)
            X_clean, y_np = X_clean[idx], y_np[idx]

        print(f"🧠 Fitting TabPFN on {len(X_clean)} samples...")
        self.model.fit(X_clean, y_np)
        return self

    def predict(self, X):
        X_clean = self._clean_input(X)
        return self.model.predict(X_clean)


# --- 4. DISTRIBUTED LOCKING UTILS ---

def get_chunk_paths(output_dir, chunk_id):
    """Definiert Dateipfade für Ergebnis und Lock."""
    filename = f"chunk_{chunk_id:06d}.parquet"
    lockname = f"chunk_{chunk_id:06d}.lock"
    return {
        "result": output_dir / filename,
        "lock": output_dir / "locks" / lockname
    }


def acquire_lock(lock_path):
    """
    Versucht atomar, ein Verzeichnis zu erstellen.
    Rückgabe: True (Lock erhalten) oder False (bereits gesperrt).
    """
    try:
        os.mkdir(lock_path)
        return True
    except FileExistsError:
        return False
    except Exception as e:
        # Bei anderen Fehlern (z.B. Permissions) lieber warnen und skippen
        print(f"⚠️ Lock Error: {e}")
        return False


def release_lock(lock_path):
    """Entfernt das Lock wieder."""
    try:
        os.rmdir(lock_path)
    except Exception:
        pass


def process_chunk(model, chunk_df, chunk_id, output_dir):
    """
    Verarbeitet einen Chunk atomar.
    """
    paths = get_chunk_paths(output_dir, chunk_id)

    # 1. Existenz-Check (Schnell)
    if paths["result"].exists():
        return False  # Schon fertig

    # 2. Lock-Versuch (Atomar)
    if not acquire_lock(paths["lock"]):
        return False  # Jemand anders rechnet gerade

    try:
        # --- CRITICAL SECTION ---

        # Metadata sichern
        meta_cols = ['snapshot_times', 'delivery_start']
        # Prüfen welche Meta-Cols da sind
        present_meta = [c for c in meta_cols if c in chunk_df.columns]

        # Vorhersage
        y_pred = model.predict(chunk_df)

        # Ergebnis DataFrame bauen
        res_df = chunk_df[present_meta].copy()
        res_df['y_pred'] = y_pred

        # Atomares Schreiben: Erst .tmp, dann rename
        tmp_path = paths["result"].with_suffix(".tmp")
        res_df.to_parquet(tmp_path, index=False)
        os.rename(tmp_path, paths["result"])

        # --- END CRITICAL SECTION ---

    except Exception as e:
        print(f"❌ Error processing chunk {chunk_id}: {e}")
        # Wir löschen den Lock, damit es ein anderer Job nochmal versuchen kann
        # (Oder wir lassen ihn, um zu zeigen 'hier ist was kaputt')
        # Hier: Löschen, retry allow.
    finally:
        release_lock(paths["lock"])

    return True


# --- 5. MAIN EXECUTION ---

def main():
    # 5.1 Setup
    out_dir = CONFIG["OUTPUT_DIR"]
    lock_dir = out_dir / "locks"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)

    print(f"🌍 Output Dir: {out_dir}")
    print(f"🔒 Lock Dir:   {lock_dir}")

    # 5.2 Daten Laden (Jeder Job macht das am Anfang einmal)
    print("📥 Loading Train Data (Context)...")
    loader = DataLoader(cfg=CONFIG, train_path=PROJECT_ROOT / "data/parquet/features/asinh1-reg/splits/train.parquet")
    X_train, y_train = loader.load_train_data(sample_n=CONFIG["HPO_SAMPLE_N"], target_col=CONFIG["TARGET_VARIABLE"],
                                              keep_id_cols=True, features_blacklist_keywords=["label_"])

    print("📥 Loading Test Data (Evaluation)...")
    # Hier laden wir ALLES, aber wir processen nur kleine Teile
    # Dank Memory-Mapping von Parquet ist das RAM-technisch oft okay,
    # solange wir nicht alles in Objekte wandeln.
    X_test, y_test = load_test_data_v2(PROJECT_ROOT / "data/parquet/features/asinh1-reg/splits/val_purged.parquet",
                                       target_col=CONFIG["TARGET_VARIABLE"], keep_id_cols=True,
                                       sample_minutes=5)
    snapshot_times, delivery_starts = X_test['snapshot_times'], X_test['delivery_start']
    X_test_clean = X_test.drop(columns=['delivery_start', 'snapshot_times'])

    # 5.3 Modell "Fitten" (In-Context Prep)
    print("🧠 Preparing TabPFN...")
    model = TabPFNRegressorWrapper(max_train_samples=CONFIG["HPO_SAMPLE_N"], device='cuda')
    model.fit(X_train, y_train)

    # 5.4 Job Loop
    chunk_size = CONFIG["CHUNK_SIZE"]
    n_samples = len(X_test)
    n_chunks = int(np.ceil(n_samples / chunk_size))

    # Liste der Chunk-IDs
    chunk_indices = list(range(n_chunks))

    # ZUFALLSSHUFFLE: Das ist der Trick, damit 20 Jobs nicht alle Chunk 0 wollen
    random.shuffle(chunk_indices)

    print(f"🚀 Starting Processing of {n_chunks} chunks (Size: {chunk_size})...")

    processed_count = 0

    # Wir nutzen TQDM, um Fortschritt dieses einen Jobs zu sehen
    pbar = tqdm(chunk_indices, desc="My Chunks", ascii=True)

    for chunk_idx in pbar:
        # Slice berechnen
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples)

        # Daten holen (Pandas Slice ist meist View oder copy-on-write, effizient)
        chunk_df = X_test.iloc[start_idx:end_idx]

        # Versuchen zu bearbeiten
        success = process_chunk(model, chunk_df, chunk_idx, out_dir)

        if success:
            processed_count += 1
            pbar.set_postfix({"processed": processed_count})

            # Garbage Collection alle paar Chunks, um RAM-Leaks in TabPFN/Torch vorzubeugen
            if processed_count % 10 == 0:
                gc.collect()

    print(f"🏁 Job Finished. I processed {processed_count} chunks.")


if __name__ == "__main__":
    main()
