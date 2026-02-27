import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# --- CONFIG ---
# Pfad zu deinen Distributed-Ergebnissen
INPUT_DIR = Path(
    "/Users/robin/PycharmProjects/Masterarbeit/data/parquet/predictions/distributed/tabpfn_reg_100k_77f").resolve()

# Pfad für die finale Datei (z.B. einen Ordner höher oder im selben)
OUTPUT_FILE = INPUT_DIR.parent / "TABPFN_100k_77f_FULL_PREDS.parquet"


def main():
    if not INPUT_DIR.exists():
        print(f"❌ Error: Directory {INPUT_DIR} does not exist.")
        return

    # 1. Alle Chunks finden
    files = sorted(list(INPUT_DIR.glob("chunk_*.parquet")))
    print(f"📦 Found {len(files)} chunk files.")

    if len(files) == 0:
        print("❌ No files found. Aborting.")
        return

    # 2. Einlesen
    dfs = []
    print("🚀 Reading chunks...")
    for f in tqdm(files, ascii=True):
        try:
            df_chunk = pd.read_parquet(f)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"⚠️ Error reading {f.name}: {e}")

    # 3. Zusammenfügen
    if not dfs:
        print("❌ No valid dataframes loaded.")
        return

    print("🔗 Concatenating...")
    full_df = pd.concat(dfs, ignore_index=True)

    initial_len = len(full_df)
    print(f"   Total rows loaded: {initial_len}")

    # 4. Sortieren (WICHTIG!)
    # Die Distributed Jobs liefen randomisiert. Wir müssen die Zeitordnung wiederherstellen.
    print("timely Sorting data...")
    if 'delivery_start' in full_df.columns and 'snapshot_times' in full_df.columns:
        full_df = full_df.sort_values(by=['delivery_start', 'snapshot_times'])
    else:
        print("⚠️ Warning: Time columns not found. Sorting skipped.")

    # 5. Check auf Duplikate (Falls Jobs doppelt liefen)
    # Wir droppen Duplikate basierend auf Zeitstempeln, falls vorhanden
    if 'snapshot_times' in full_df.columns:
        full_df = full_df.drop_duplicates(subset=['snapshot_times', 'delivery_start'], keep='last')
        if len(full_df) < initial_len:
            print(f"🧹 Removed {initial_len - len(full_df)} duplicates.")

    # 6. Check auf NaNs in Prediction (Falls Timeout-Skript NaNs hinterlassen hat)
    if 'y_pred' in full_df.columns:
        nan_count = full_df['y_pred'].isna().sum()
        if nan_count > 0:
            print(f"⚠️ ACHTUNG: {nan_count} NaNs in 'y_pred' gefunden!")
            # Option: Fillna mit 0 oder Mean, oder so lassen
            # full_df['y_pred'] = full_df['y_pred'].fillna(0)
        else:
            print("✅ No NaNs in predictions.")

    # 7. Speichern
    print(f"💾 Saving to {OUTPUT_FILE}...")
    full_df.to_parquet(OUTPUT_FILE, index=False)

    print("\n✅ DONE!")
    print(f"   Final Shape: {full_df.shape}")
    print(f"   File saved at: {OUTPUT_FILE}")

    # Optional: Aufräumen
    # print("To clean up individual chunks, run: rm -rf " + str(INPUT_DIR))


if __name__ == "__main__":
    main()
