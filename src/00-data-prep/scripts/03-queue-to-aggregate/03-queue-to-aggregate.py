import pandas as pd
from pathlib import Path
from tqdm import tqdm
import collections
import numpy as np

# --- KONFIGURATION ---
INPUT_DIRECTORY = Path("../../../data/parquet/02-cleaned-batches")
OUTPUT_DIRECTORY = Path("../../../data/parquet/03-aggregated-batches")

MIN_UNIQUE_LEVELS_FOR_CENSORED = 11
TARGET_LEVELS = 10


# ---------------------

def main():
    """
    Hauptfunktion, die den finalen, korrekten Algorithmus zur Strukturierung
    der Orderbuch-Daten in einer optimierten, vektorisierten Pipeline umsetzt.
    """

    # logge Konfiguration
    print("=" * 50)
    print("--- KONFIGURATION ---")
    print(f"Startzeitpunkt: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Eingabe-Verzeichnis (bereinigte Daten): {INPUT_DIRECTORY}")
    print(f"Ausgabe-Verzeichnis (strukturierte Daten): {OUTPUT_DIRECTORY}")
    print(f"Minimale einzigartige Preis-Level für zensierte Seiten: {MIN_UNIQUE_LEVELS_FOR_CENSORED}")
    print(f"Ziel-Level im finalen Orderbuch: {TARGET_LEVELS}")
    print("=" * 50 + "\n")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(list(INPUT_DIRECTORY.glob("*.parquet")))
    print(f"Starte korrekte, optimierte Verarbeitung von {len(parquet_files)} Parquet-Dateien...")

    global_stats = collections.defaultdict(int)
    COLS_TO_LOAD = ['snapshot_times', 'delivery_start', 'side', 'price', 'quantity', 'orderbook_level']

    for file_path in tqdm(parquet_files, desc="Verarbeite Batches"):
        try:
            df = pd.read_parquet(file_path, columns=COLS_TO_LOAD)
            if df.empty:
                (OUTPUT_DIRECTORY / file_path.name).touch();
                continue

            # --- Schritt 1: Snapshot-Metadaten erstellen (Zensurierung & liquide Tiefe) ---

            # Erstelle eine Tabelle mit einer Zeile pro Snapshot für die Filter-Logik
            snapshot_keys = ['snapshot_times', 'delivery_start']

            # KORREKTE ZÄHLUNG von einzigartigen Snapshots
            total_snapshots_in_batch = df[snapshot_keys].drop_duplicates().shape[0]

            snapshot_summary = df[snapshot_keys].drop_duplicates()

            # a) Zensurierungs-Status ermitteln
            max_levels = df.groupby(snapshot_keys + ['side'])['orderbook_level'].max().unstack()
            snapshot_summary = snapshot_summary.merge(max_levels, on=snapshot_keys, how='left').rename(
                columns={'BID': 'max_level_bid', 'ASK': 'max_level_ask'}).fillna(0)
            snapshot_summary['is_bid_censored'] = snapshot_summary['max_level_bid'] == 20
            snapshot_summary['is_ask_censored'] = snapshot_summary['max_level_ask'] == 20

            # b) Liquide Tiefe ermitteln
            df_with_volume = df[df['quantity'] > 0]
            liquid_depths = df_with_volume.groupby(snapshot_keys + ['side'])['price'].nunique().unstack()
            snapshot_summary = snapshot_summary.merge(liquid_depths, on=snapshot_keys, how='left').rename(
                columns={'BID': 'liquid_depth_bid', 'ASK': 'liquid_depth_ask'}).fillna(0)

            # --- Schritt 2: Invalide Snapshots identifizieren & filtern ---
            is_bid_invalid = (snapshot_summary['is_bid_censored']) & (
                    snapshot_summary['liquid_depth_bid'] < MIN_UNIQUE_LEVELS_FOR_CENSORED)
            is_ask_invalid = (snapshot_summary['is_ask_censored']) & (
                    snapshot_summary['liquid_depth_ask'] < MIN_UNIQUE_LEVELS_FOR_CENSORED)

            invalid_snapshots = snapshot_summary[is_bid_invalid | is_ask_invalid]

            # Behalte nur die gültigen Snapshots im Original-DataFrame
            if not invalid_snapshots.empty:
                df = df.merge(invalid_snapshots[snapshot_keys], on=snapshot_keys, how='left', indicator=True)
                df_surviving = df[df['_merge'] == 'left_only'].drop(columns=['_merge']).copy()
            else:
                df_surviving = df.copy()

            # --- Schritt 3: Aggregation & Anti-Zensurierung ---
            if not df_surviving.empty:
                # Merge Zensurierungs-Info für Anti-Zensurierung
                df_surviving = df_surviving.merge(
                    snapshot_summary[snapshot_keys + ['is_bid_censored', 'is_ask_censored']], on=snapshot_keys,
                    how='left')

                agg_df = df_surviving.groupby(snapshot_keys + ['side', 'price', 'is_bid_censored', 'is_ask_censored'],
                                              as_index=False)['quantity'].sum()

                # Anti-Zensurierung mit .drop_duplicates statt transform (schneller)
                bids_df = agg_df[agg_df['side'] == 'BID'].sort_values('price', ascending=False).copy()
                asks_df = agg_df[agg_df['side'] == 'ASK'].sort_values('price', ascending=True).copy()

                bids_to_drop = bids_df[(bids_df['is_bid_censored'])].drop_duplicates(subset=snapshot_keys, keep='last')
                asks_to_drop = asks_df[(asks_df['is_ask_censored'])].drop_duplicates(subset=snapshot_keys, keep='last')

                bids_df = bids_df.merge(bids_to_drop, how='left', indicator=True).query('_merge == "left_only"').drop(
                    columns=['_merge'])
                asks_df = asks_df.merge(asks_to_drop, how='left', indicator=True).query('_merge == "left_only"').drop(
                    columns=['_merge'])

                agg_df_cleaned = pd.concat([bids_df, asks_df], ignore_index=True)

                # --- Schritt 4: Level-Zuweisung, Kürzung & Padding ---
                bids = agg_df_cleaned[agg_df_cleaned['side'] == 'BID'].sort_values(
                    ['snapshot_times', 'delivery_start', 'price'], ascending=[True, True, False])
                asks = agg_df_cleaned[agg_df_cleaned['side'] == 'ASK'].sort_values(
                    ['snapshot_times', 'delivery_start', 'price'], ascending=[True, True, True])

                bids['orderbook_level'] = bids.groupby(snapshot_keys).cumcount() + 1
                asks['orderbook_level'] = asks.groupby(snapshot_keys).cumcount() + 1

                processed_df = pd.concat([bids, asks], ignore_index=True)
                processed_df = processed_df[processed_df['orderbook_level'] <= TARGET_LEVELS]

                unique_snapshots = processed_df[snapshot_keys].drop_duplicates()
                template = pd.DataFrame({'side': ['BID'] * TARGET_LEVELS + ['ASK'] * TARGET_LEVELS,
                                         'orderbook_level': list(range(1, TARGET_LEVELS + 1)) * 2})
                full_template = unique_snapshots.merge(template, how='cross')
                final_df = pd.merge(full_template, processed_df, on=snapshot_keys + ['side', 'orderbook_level'],
                                    how='left')
            else:
                final_df = pd.DataFrame()

            # --- Schritt 5: Speichern und Logging ---
            output_file_path = OUTPUT_DIRECTORY / file_path.name
            final_df.to_parquet(output_file_path, index=False)

            # Per-File Logging
            survived_snapshots_count = final_df[snapshot_keys].drop_duplicates().shape[0] if not final_df.empty else 0
            dropped_snapshots_count = total_snapshots_in_batch - survived_snapshots_count

            tqdm.write(f"\n--- Verarbeitung für: {file_path.name} ---")
            tqdm.write(f"  Snapshots im Batch: {total_snapshots_in_batch:,}")
            tqdm.write(f"  Snapshots verworfen (dünn & zensiert): {dropped_snapshots_count:,}")
            tqdm.write(f"  Snapshots verarbeitet: {survived_snapshots_count:,}")
            tqdm.write(f"  Resultierende Zeilen (L10): {len(final_df):,}")

            global_stats['rows_in'] += len(df)
            global_stats['rows_out'] += len(final_df)
            global_stats['snapshots_in'] += total_snapshots_in_batch
            global_stats['snapshots_dropped'] += dropped_snapshots_count

        except Exception as e:
            tqdm.write(f"\nFEHLER bei der Verarbeitung von {file_path.name}: {e}")
            continue

    # --- Finale Zusammenfassung ---
    print("\n" + "=" * 50)
    print("--- FINALE ZUSAMMENFASSUNG DER STRUKTURIERUNG ---")
    print("=" * 50 + "\n")
    print(f"Gesamtzahl verarbeiteter Roh-Zeilen: {global_stats['rows_in']:,}")
    print(f"Gesamtzahl resultierender strukturierter Zeilen: {global_stats['rows_out']:,}")
    print("-" * 30)
    print(f"Gesamtzahl der Snapshots in Rohdaten: {global_stats['snapshots_in']:,}")
    print(f"Gesamtzahl der verworfenen Snapshots: {global_stats['snapshots_dropped']:,}")

    snapshots_survived = global_stats['snapshots_in'] - global_stats['snapshots_dropped']
    percent_survived = (snapshots_survived / global_stats['snapshots_in']) * 100 if global_stats[
                                                                                        'snapshots_in'] > 0 else 0
    print(f"Gesamtzahl der finalen Snapshots: {snapshots_survived:,} ({percent_survived:.2f}%)")

    print("\n--- Prozess abgeschlossen ---")


if __name__ == "__main__":
    main()
