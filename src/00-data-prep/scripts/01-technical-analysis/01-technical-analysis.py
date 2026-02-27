import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import collections
from typing import Dict, Any, List

# --- KONFIGURATION ---
# Pfad zum Verzeichnis mit den Parquet-Dateien
DATA_DIRECTORY = Path("../../../data/parquet/00-0-raw")

# Grenzwerte für forensische Prüfungen
PRICE_UPPER_LIMIT = 9999  # technische Grenze
PRICE_LOWER_LIMIT = -9999  # technische Grenze
QUANTITY_UPPER_LIMIT = 1000
ORDERBOOK_LEVEL_MAX = 20

# Konfiguration für deskriptive Analyse und Plots
SAMPLE_SIZE_PER_FILE = 500  # Wichtig für die Genauigkeit des Median/der Quantile
HIST_BINS_PRICE = 100
HIST_RANGE_PRICE = (-500, 1000)
HIST_BINS_QUANTITY = 100
HIST_RANGE_QUANTITY = (0, 100)


# ---------------------

def process_single_file(file_path: Path) -> Dict[str, Any]:
    """
    Lädt alle Spalten einer Parquet-Datei und führt eine umfassende Analyse durch.
    Gibt alle Ergebnisse in einem strukturierten Dictionary zurück.
    """
    stats = collections.defaultdict(int)
    stats['file_name'] = file_path.name

    # Lade alle Spalten, die in der Datei vorhanden sind
    df = pd.read_parquet(file_path)

    # --- Forensische Analyse ---
    stats['total_rows'] = len(df)
    stats['null_counts'] = df.isnull().sum()

    # Robuste Konvertierung der Zeitstempel
    time_cols = ['snapshot_times', 'delivery_start', 'creation_time']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    stats['negative_quantity'] = (df['quantity'] < 0).sum()

    if 'snapshot_times' in df.columns and 'delivery_start' in df.columns:
        stats['post_delivery_snapshot'] = (df['snapshot_times'] > df['delivery_start']).sum()

    if 'creation_time' in df.columns and 'snapshot_times' in df.columns:
        temp_df_created = df.dropna(subset=['creation_time', 'snapshot_times'])
        stats['snapshot_before_creation'] = (temp_df_created['snapshot_times'] < temp_df_created['creation_time']).sum()

    stats['price_outside_limits'] = ((df['price'] > PRICE_UPPER_LIMIT) | (df['price'] < PRICE_LOWER_LIMIT)).sum()
    stats['level_outside_range'] = ((df['orderbook_level'] < 1) | (df['orderbook_level'] > ORDERBOOK_LEVEL_MAX)).sum()
    stats['unrealistic_quantity'] = (df['quantity'] > QUANTITY_UPPER_LIMIT).sum()

    # --- Deskriptive Analyse & Komponenten für globale Berechnung ---
    stats['descriptive_stats'] = df[['price', 'quantity']].describe()
    stats['sum_of_squares'] = (df[['price', 'quantity']] ** 2).sum()

    # --- Histogramm-Daten & Stichprobe ---
    stats['price_hist'], _ = np.histogram(df['price'].dropna(), bins=HIST_BINS_PRICE, range=HIST_RANGE_PRICE)
    stats['quantity_hist'], _ = np.histogram(df['quantity'].dropna(), bins=HIST_BINS_QUANTITY,
                                             range=HIST_RANGE_QUANTITY)

    n_samples = min(SAMPLE_SIZE_PER_FILE, len(df))
    stats['sampled_df'] = df[['price', 'quantity']].sample(n=n_samples,
                                                           random_state=42) if n_samples > 0 else pd.DataFrame()

    if 'snapshot_times' in df.columns:
        stats['unique_snapshots'] = df['snapshot_times'].dropna().unique()
    else:
        stats['unique_snapshots'] = []

    return stats


def print_final_summary(stats: Dict[str, Any], all_snapshots: set, sampled_dfs: List[pd.DataFrame]):
    """ Gibt die finale, aggregierte Zusammenfassung mit allen Details aus. """
    print("\n" + "=" * 50)
    print("--- FINALE FORENSISCHE GESAMTANALYSE ---")
    print("=" * 50 + "\n")

    print(f"Gesamtzahl der analysierten Order-Einträge: {stats['total_rows']:,}")

    print("\n--- Gesamt-Nullwerte pro Spalte ---")
    if 'null_counts' in stats and not stats['null_counts'].empty:
        print(stats['null_counts'].astype(int).to_string())
    else:
        print("Keine Nullwerte gefunden oder gezählt.")

    print("\n--- Zusammenfassung der erkannten Irregularitäten ---")
    print(f"Fehlende Quantitäts-Werte (NaN): {stats['null_counts'].get('quantity', 0):,}")
    print(f"Orders mit negativer Quantität (Fehler): {stats['negative_quantity']:,}")
    print(
        f"Records mit unrealistisch großer Quantität (> {QUANTITY_UPPER_LIMIT} MWh): {stats['unrealistic_quantity']:,}")
    print("-" * 20)
    print(f"Records mit Snapshot nach Beginn der Lieferzeit: {stats['post_delivery_snapshot']:,}")
    print(f"Records mit Snapshot vor Erstellzeit der Order: {stats['snapshot_before_creation']:,}")
    print(
        f"Records mit Preis außerhalb der Grenzen ({PRICE_LOWER_LIMIT}/{PRICE_UPPER_LIMIT} EUR): {stats['price_outside_limits']:,}")
    print(
        f"Records mit Orderbook-Level außerhalb des Bereichs (1-{ORDERBOOK_LEVEL_MAX}): {stats['level_outside_range']:,}")

    print("\n--- Kumulative Deskriptive Statistik ---")
    if stats['cumulative_count']['price'] > 0:
        variance = (stats['cumulative_sum_sq'] / stats['cumulative_count']) - (stats['cumulative_mean'] ** 2)
        std_dev = np.sqrt(variance.abs())
    else:
        std_dev = pd.Series({'price': np.nan, 'quantity': np.nan})

    if sampled_dfs:
        final_sample_df = pd.concat(sampled_dfs, ignore_index=True)
        quantiles = final_sample_df.quantile([0.25, 0.5, 0.75])
        q25, median, q75 = quantiles.loc[0.25], quantiles.loc[0.50], quantiles.loc[0.75]
    else:
        q25 = median = q75 = pd.Series({'price': np.nan, 'quantity': np.nan})

    cumulative_desc = pd.DataFrame({
        'count': stats['cumulative_count'],
        'mean': stats['cumulative_mean'],
        'std': std_dev,
        'min': stats['cumulative_min'],
        '25%': q25,
        '50% (median)': median,
        '75%': q75,
        'max': stats['cumulative_max']
    })
    print(cumulative_desc.to_string())
    print(
        "* Hinweis: 'std' wurde exakt berechnet. Quantile ('25%', '50%', '75%') sind Näherungen basierend auf einer kombinierten Stichprobe.")

    if all_snapshots:
        unique_snapshots = pd.Series(list(all_snapshots)).sort_values()
        gaps = unique_snapshots.diff().dt.total_seconds()
        print("\n--- Analyse der Lücken in `snapshot_times` ---")
        print(f"Anzahl einzigartiger Snapshots gefunden: {len(unique_snapshots):,}")
        print(f"Maximale Lücke zwischen zwei Snapshots: {gaps.max():,.2f} Sekunden")
        print(f"Anzahl Lücken > 300 Sekunden (5 Min): {(gaps > 300).sum():,}")


def main():
    """ Hauptfunktion zur Steuerung des gesamten Analyseprozesses. """
    if not DATA_DIRECTORY.is_dir():
        print(f"Fehler: Das Verzeichnis '{DATA_DIRECTORY}' wurde nicht gefunden.")
        return
    parquet_files = sorted(list(DATA_DIRECTORY.glob("*.parquet")))
    if not parquet_files:
        print(f"Fehler: Keine .parquet-Dateien im Verzeichnis '{DATA_DIRECTORY}' gefunden.")
        return

    print(f"Umfassende Analyse von {len(parquet_files)} Parquet-Dateien gestartet...")

    # Initialisierung der globalen Aggregatoren
    global_stats = collections.defaultdict(int)
    global_stats['null_counts'] = pd.Series(dtype=int)
    global_stats['cumulative_count'] = pd.Series({'price': 0.0, 'quantity': 0.0})
    global_stats['cumulative_mean'] = pd.Series({'price': 0.0, 'quantity': 0.0})
    global_stats['cumulative_sum_sq'] = pd.Series({'price': 0.0, 'quantity': 0.0})
    global_stats['cumulative_min'] = pd.Series({'price': np.inf, 'quantity': np.inf})
    global_stats['cumulative_max'] = pd.Series({'price': -np.inf, 'quantity': -np.inf})

    price_hist_total = np.zeros(HIST_BINS_PRICE)
    quantity_hist_total = np.zeros(HIST_BINS_QUANTITY)
    all_sampled_dfs = []
    all_snapshot_times = set()

    for file_path in tqdm(parquet_files, desc="Verarbeite Dateien"):
        try:
            per_file_results = process_single_file(file_path)

            # --- Detailliertes Per-File Logging ---
            tqdm.write(f"\n" + "=" * 20 + f" Analyse für: {per_file_results['file_name']} " + "=" * 20)

            tqdm.write(f"Verarbeitete Zeilen: {per_file_results['total_rows']:,}")

            tqdm.write("\n--- Nullwerte pro Spalte ---")
            tqdm.write(per_file_results['null_counts'].to_string())

            tqdm.write("\n--- Forensische Metriken ---")
            tqdm.write(f"Negative Quantität (Fehler): {per_file_results['negative_quantity']:,}")
            tqdm.write(f"Snapshot nach Lieferung: {per_file_results['post_delivery_snapshot']:,}")
            tqdm.write(f"Snapshot vor Erstellung: {per_file_results['snapshot_before_creation']:,}")
            tqdm.write(f"Preis außerhalb Grenzen: {per_file_results['price_outside_limits']:,}")
            tqdm.write(f"Level außerhalb Bereich: {per_file_results['level_outside_range']:,}")
            tqdm.write(f"Unrealistische Quantität: {per_file_results['unrealistic_quantity']:,}")

            tqdm.write("\n--- Deskriptive Statistik der Datei ---")
            tqdm.write(per_file_results['descriptive_stats'].to_string())
            tqdm.write("=" * (42 + len(per_file_results['file_name'])))

            # --- Update der globalen Aggregatoren ---
            forensic_keys = ['total_rows', 'negative_quantity', 'post_delivery_snapshot',
                             'snapshot_before_creation', 'price_outside_limits', 'level_outside_range',
                             'unrealistic_quantity']
            for key in forensic_keys:
                global_stats[key] += per_file_results[key]

            global_stats['null_counts'] = global_stats['null_counts'].add(per_file_results['null_counts'], fill_value=0)

            desc = per_file_results['descriptive_stats']
            count = desc.loc['count']
            if (global_stats['cumulative_count']['price'] + count['price']) > 0:
                new_total_count = global_stats['cumulative_count'] + count
                global_stats['cumulative_mean'] = ((global_stats['cumulative_mean'] * global_stats[
                    'cumulative_count']) + (desc.loc['mean'] * count)) / new_total_count
                global_stats['cumulative_count'] = new_total_count

            global_stats['cumulative_sum_sq'] += per_file_results['sum_of_squares']
            global_stats['cumulative_min'] = np.minimum(global_stats['cumulative_min'], desc.loc['min'])
            global_stats['cumulative_max'] = np.maximum(global_stats['cumulative_max'], desc.loc['max'])

            price_hist_total += per_file_results['price_hist']
            quantity_hist_total += per_file_results['quantity_hist']
            if not per_file_results['sampled_df'].empty:
                all_sampled_dfs.append(per_file_results['sampled_df'])
            all_snapshot_times.update(per_file_results['unique_snapshots'])

        except Exception as e:
            tqdm.write(f"\nFEHLER beim Verarbeiten der Datei {file_path}: {e}")
            continue

    print_final_summary(global_stats, all_snapshot_times, all_sampled_dfs)

    print("\n--- Analyse abgeschlossen ---")


if __name__ == "__main__":
    main()
