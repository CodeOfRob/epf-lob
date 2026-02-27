from pathlib import Path
import os

# --- Pfade & Umgebung ---
# automatische Erkennung: Cluster oder Lokal
HOME = os.getenv("HOME", ".")
if "to65jevo" in HOME:  # Indikator für Cluster-Pfads
    BASEPATH = Path(HOME) / "epf-with-ml-on-orderbooks"
else:
    # Fallback für lokal oder wenn HOME anders gesetzt ist
    BASEPATH = Path("/Users/robin/PycharmProjects/Masterarbeit")

# --- Metadaten ---
N_TRIALS = 100
N_CV_SPLITS = 5
RANDOM_STATE = 42
TARGET_VARIABLE = "label_5min"

# NEU: Zielgröße für HPO-Trials (z.B. 250.000 Samples)
# Setze auf None, um die vollen 13,5 Mio. zu nutzen (nur für schnelle Modelle empfohlen)
HPO_SAMPLE_N = 250000

# --- Feature Filterung ---
# Alles, was eines dieser Wörter enthält, wird NICHT als Feature genutzt.
FEATURE_BLACKLIST_KEYWORDS = [
    "delivery_start",  # Key
    "snapshot_times",  # Key
    "label",  # Target (alle Varianten)
    "normalized_price",  # Altes Target
    "product_id"
]

DATA_DIR = BASEPATH / f"data/parquet/features/widened-min-periods/splits"

PATHS = {
    "train_file": DATA_DIR / "test.parquet",
    "output_dir": BASEPATH / "src/hpo/hpo_results",  # Basis-Output
    "optuna_db": f"sqlite:///{BASEPATH}/src/hpo/optuna/test.db"
}


def get_model(params):
    raise NotImplementedError("Muss im Child implementiert werden")


def get_search_space(trial):
    raise NotImplementedError("Muss im Child implementiert werden")
