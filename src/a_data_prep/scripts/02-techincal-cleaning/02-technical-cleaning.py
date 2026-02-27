import pandas as pd
from pathlib import Path
from tqdm import tqdm
import collections
import numpy as np

# --- KONFIGURATION ---
# Eingabe: Die rohesten Produkt-Batches
INPUT_DIRECTORY = Path("../../../data/parquet/00-1-product-batches")
# Ausgabe: Vollständig bereinigte Daten (technisch & logisch)
OUTPUT_DIRECTORY = Path("../../../data/parquet/02-cleaned-batches")

# Parameter für die technische Bereinigung
PRICE_UPPER_LIMIT = 9999
PRICE_LOWER_LIMIT = -9999
QUANTITY_UPPER_LIMIT = 1000
ORDERBOOK_LEVEL_MAX = 20


# ---------------------

def repair_crossed_books(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Identifiziert und repariert Crossed Books in einem bereits technisch bereinigten DataFrame.
    """
    stats = collections.defaultdict(int)
    snapshot_keys = ['snapshot_times', 'delivery_start']
    if df.empty:
        stats['num_crossed_snapshots'] = 0
        stats['rows_removed_crossing'] = 0
        return pd.DataFrame(), stats

    best_bids = df[df['side'] == 'BID'].groupby(snapshot_keys)['price'].max().rename('best_bid')
    best_asks = df[df['side'] == 'ASK'].groupby(snapshot_keys)['price'].min().rename('best_ask')

    snapshot_meta = df[snapshot_keys].drop_duplicates().merge(best_bids, on=snapshot_keys, how='left').merge(best_asks,
                                                                                                             on=snapshot_keys,
                                                                                                             how='left')
    crossed_snapshots_meta = snapshot_meta[snapshot_meta['best_bid'] >= snapshot_meta['best_ask']].copy()
    stats['num_crossed_snapshots'] = len(crossed_snapshots_meta)

    if crossed_snapshots_meta.empty:
        stats['rows_removed_crossing'] = 0
        return df, stats

    df_merged = df.merge(crossed_snapshots_meta[snapshot_keys + ['best_bid', 'best_ask']], on=snapshot_keys, how='left')

    is_bad_bid = (df_merged['side'] == 'BID') & (df_merged['price'] >= df_merged['best_ask'])
    is_bad_ask = (df_merged['side'] == 'ASK') & (df_merged['price'] <= df_merged['best_bid'])
    rows_to_remove_mask = is_bad_bid | is_bad_ask

    stats['rows_removed_crossing'] = rows_to_remove_mask.sum()
    final_df = df_merged[~rows_to_remove_mask].copy().drop(columns=['best_bid', 'best_ask'])

    return final_df, stats


def clean_and_repair_file(input_path: Path, output_path: Path) -> collections.defaultdict:
    """
    Führt die gesamte Bereinigungspipeline für eine einzelne Datei aus.
    """
    stats = collections.defaultdict(int)
    df = pd.read_parquet(input_path)
    stats['rows_in'] = len(df)

    if df.empty:
        df.to_parquet(output_path, index=False)
        return stats

    # --- PHASE 1: TECHNISCHE BEREINIGUNG ---
    df.drop(columns=['market_area', 'exchange', 'currency', 'type'], inplace=True, errors='ignore')

    rows_before_malformed = len(df)
    critical_cols = ['price', 'quantity', 'order_id', 'product_id', 'snapshot_times', 'creation_time']
    df.dropna(subset=critical_cols, inplace=True)
    stats['dropped_malformed_critical'] = rows_before_malformed - len(df)

    rows_before_time = len(df)
    time_cols = ['snapshot_times', 'delivery_start', 'creation_time']
    for col in time_cols: df[col] = pd.to_datetime(df[col], errors='coerce')
    df.dropna(subset=time_cols, inplace=True)
    stats['dropped_malformed_time'] = rows_before_time - len(df)

    valid_mask = (
            (df['snapshot_times'] >= df['creation_time']) &
            (df['snapshot_times'] <= df['delivery_start']) &
            (df['price'] >= PRICE_LOWER_LIMIT) & (df['price'] <= PRICE_UPPER_LIMIT) &
            (df['quantity'] >= 0) &
            (df['orderbook_level'] >= 1) & (df['orderbook_level'] <= ORDERBOOK_LEVEL_MAX) &
            (df['quantity'] <= QUANTITY_UPPER_LIMIT) &
            ~((df['price'].notna()) & (df['quantity'] <= 1e-9))
    )
    rows_before_inconsistent = len(df)
    df_technically_cleaned = df[valid_mask].copy()
    stats['dropped_inconsistent_technical'] = rows_before_inconsistent - len(df_technically_cleaned)

    # --- PHASE 2: LOGISCHE BEREINIGUNG (CROSSED BOOK REPAIR) ---
    df_repaired, repair_stats = repair_crossed_books(df_technically_cleaned)
    stats.update(repair_stats)

    # --- FINALE AKTIONEN ---
    stats['rows_out'] = len(df_repaired)
    stats['total_dropped'] = stats['rows_in'] - stats['rows_out']
    df_repaired.to_parquet(output_path, index=False)

    return stats


def main():
    """Hauptfunktion zur Steuerung des gesamten Bereinigungsprozesses."""
    # --- Logge Konfiguration ---
    print("=" * 50)
    print("--- KONFIGURATION & START ---")
    print(f"Startzeitpunkt: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Eingabe-Verzeichnis: {INPUT_DIRECTORY}")
    print(f"Ausgabe-Verzeichnis: {OUTPUT_DIRECTORY}")
    print(f"Preisgrenzen: [{PRICE_LOWER_LIMIT}, {PRICE_UPPER_LIMIT}]")
    print(f"Mengen-Obergrenze: {QUANTITY_UPPER_LIMIT}")
    print(f"Maximales Orderbuch-Level: {ORDERBOOK_LEVEL_MAX}")
    print("=" * 50 + "\n")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(list(INPUT_DIRECTORY.glob("*.parquet")))
    if not parquet_files:
        print(f"Fehler: Keine Dateien in '{INPUT_DIRECTORY}' gefunden.")
        return

    print(f"Starte vollständige Bereinigung für {len(parquet_files)} Dateien...")
    global_stats = collections.defaultdict(int)

    for file_path in tqdm(parquet_files, desc="Bereinige Batches"):
        try:
            output_file_path = OUTPUT_DIRECTORY / file_path.name
            per_file_stats = clean_and_repair_file(file_path, output_file_path)

            # --- Per-File Logging ---
            tqdm.write("\n" + "-" * 50)
            tqdm.write(f"Verarbeitung abgeschlossen für: {file_path.name}")
            tqdm.write("-" * 50)
            tqdm.write(f"  Zeilen Eingabe: {per_file_stats['rows_in']:,}")
            tqdm.write("  Phase 1 (Technische Bereinigung):")
            tqdm.write(f"    - Entfernt (fehlende krit. Spalten): {per_file_stats['dropped_malformed_critical']:,}")
            tqdm.write(f"    - Entfernt (fehlende Zeitstempel): {per_file_stats['dropped_malformed_time']:,}")
            tqdm.write(f"    - Entfernt (techn. inkonsistent): {per_file_stats['dropped_inconsistent_technical']:,}")
            tqdm.write("  Phase 2 (Logische Bereinigung):")
            tqdm.write(f"    - Gefundene Crossed-Book-Snapshots: {per_file_stats['num_crossed_snapshots']:,}")
            tqdm.write(f"    - Entfernte Crossing-Orders: {per_file_stats['rows_removed_crossing']:,}")
            tqdm.write("-" * 50)
            tqdm.write(f"  Zeilen Gesamt entfernt: {per_file_stats['total_dropped']:,}")
            tqdm.write(f"  Zeilen Ausgabe: {per_file_stats['rows_out']:,}")

            for key, value in per_file_stats.items():
                global_stats[key] += value

        except Exception as e:
            tqdm.write(f"\nFEHLER bei der Verarbeitung von {file_path}: {e}")
            continue

    # --- Finale Zusammenfassung ---
    print("\n" + "=" * 50)
    print("--- FINALE ZUSAMMENFASSUNG DER GESAMTEN BEREINIGUNG ---")
    print("=" * 50 + "\n")
    print(f"Gesamtzahl der Zeilen in Rohdaten: {global_stats['rows_in']:,}")
    print("-" * 30)
    print("Entfernt in Phase 1 (Techn. Bereinigung):")
    print(f"  - Fehlende kritische Werte: {global_stats['dropped_malformed_critical']:,}")
    print(f"  - Fehlende Zeitstempel: {global_stats['dropped_malformed_time']:,}")
    print(f"  - Technische Inkonsistenzen: {global_stats['dropped_inconsistent_technical']:,}")
    print("-" * 30)
    print("Entfernt in Phase 2 (Log. Bereinigung):")
    print(f"  - Gefundene Crossed-Book-Snapshots: {global_stats['total_crossed_snapshots']:,}")
    print(f"  - Entfernte Crossing-Orders: {global_stats['rows_removed_crossing']:,}")
    print("-" * 30)
    print(f"Gesamtzahl der entfernten Zeilen: {global_stats['total_dropped']:,}")
    print(f"Gesamtzahl der Zeilen in final bereinigten Daten: {global_stats['rows_out']:,}")

    if global_stats['rows_in'] > 0:
        percent_retained = (global_stats['rows_out'] / global_stats['rows_in']) * 100
        print(f"\nAnteil der beibehaltenen Daten: {percent_retained:.2f}%")

    print("\n--- Prozess abgeschlossen ---")


if __name__ == "__main__":
    main()
