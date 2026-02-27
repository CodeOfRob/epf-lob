import pandas as pd
from pathlib import Path
from tqdm import tqdm
import collections

# --- KONFIGURATION ---
# Pfad zum Verzeichnis mit den rohen, "langen" Parquet-Dateien
INPUT_DIRECTORY = Path("/data/parquet/00-0-raw")
# Wie viele Beispiele pro Anomalie sollen gespeichert werden?
MAX_EXAMPLES_TO_STORE = 5


# ---------------------

def audit_raw_files(directory: Path):
    """
    Durchsucht rohe Parquet-Dateien nach spezifischen Anomalien, protokolliert die
    Funde pro Datei und gibt eine finale Zusammenfassung mit konkreten Beispielen aus.
    """
    if not directory.is_dir():
        print(f"Fehler: Das Eingabeverzeichnis '{directory}' wurde nicht gefunden.")
        return

    parquet_files = sorted(list(directory.glob("*.parquet")))
    if not parquet_files:
        print(f"Fehler: Keine .parquet-Dateien im Verzeichnis '{directory}' gefunden.")
        return

    print(f"Starte forensisches Audit von {len(parquet_files)} Parquet-Dateien...")

    # Initialisiere globale Zähler und Speicher für Beweise
    global_anomaly_counts = collections.defaultdict(int)
    examples = collections.defaultdict(list)

    for file_path in tqdm(parquet_files, desc="Auditiere Dateien"):
        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                continue

            per_file_counts = collections.defaultdict(int)
            per_file_examples = collections.defaultdict(list)

            grouped = df.groupby(['snapshot_times', 'product_id'])

            for (snapshot_time, product_id), snapshot_df in grouped:

                bids = snapshot_df[snapshot_df['side'] == 'BID'].sort_values('orderbook_level')
                asks = snapshot_df[snapshot_df['side'] == 'ASK'].sort_values('orderbook_level')

                # --- Anomalie 1: Gleiche Preise auf unterschiedlichen Levels ---
                if bids['price'].duplicated().any() or asks['price'].duplicated().any():
                    per_file_counts['duplicate_prices_on_side'] += 1
                    if len(examples['duplicate_prices_on_side']) < MAX_EXAMPLES_TO_STORE:
                        examples['duplicate_prices_on_side'].append((file_path.name, snapshot_time, product_id))
                    if len(per_file_examples['duplicate_prices_on_side']) < MAX_EXAMPLES_TO_STORE:
                        per_file_examples['duplicate_prices_on_side'].append((snapshot_time, product_id))

                # --- Anomalie 3 & 4: Zero-Quantity Orders ---
                if (snapshot_df['quantity'] == 0).any():
                    per_file_counts['zero_quantity_orders'] += 1
                    if len(examples['zero_quantity_orders']) < MAX_EXAMPLES_TO_STORE:
                        examples['zero_quantity_orders'].append((file_path.name, snapshot_time, product_id))
                    if len(per_file_examples['zero_quantity_orders']) < MAX_EXAMPLES_TO_STORE:
                        per_file_examples['zero_quantity_orders'].append((snapshot_time, product_id))

                    is_sandwiched = False
                    for side_df in [bids, asks]:
                        if len(side_df) >= 3:
                            q = side_df['quantity'].values
                            for i in range(1, len(q) - 1):
                                if q[i - 1] > 0 and q[i] == 0 and q[i + 1] > 0:
                                    is_sandwiched = True
                                    break
                        if is_sandwiched: break
                    if is_sandwiched:
                        per_file_counts['sandwiched_zero_quantity'] += 1
                        if len(examples['sandwiched_zero_quantity']) < MAX_EXAMPLES_TO_STORE:
                            examples['sandwiched_zero_quantity'].append((file_path.name, snapshot_time, product_id))
                        if len(per_file_examples['sandwiched_zero_quantity']) < MAX_EXAMPLES_TO_STORE:
                            per_file_examples['sandwiched_zero_quantity'].append((snapshot_time, product_id))

                # --- Anomalie 5: Crossed/Locked Books (ANGEPASST) ---
                if not bids.empty and not asks.empty:
                    best_bid = bids.iloc[0]
                    best_ask = asks.iloc[0]

                    # *** NEUE BEDINGUNG: Prüfe nur, wenn beide Seiten Liquidität haben ***
                    if best_bid['quantity'] > 0 and best_ask['quantity'] > 0:
                        if best_bid['price'] >= best_ask['price']:
                            per_file_counts['crossed_locked_books'] += 1
                            if len(examples['crossed_locked_books']) < MAX_EXAMPLES_TO_STORE:
                                examples['crossed_locked_books'].append((file_path.name, snapshot_time, product_id))
                            if len(per_file_examples['crossed_locked_books']) < MAX_EXAMPLES_TO_STORE:
                                per_file_examples['crossed_locked_books'].append((snapshot_time, product_id))

            # --- Per-File Logging ---
            tqdm.write(f"\n--- Audit-Ergebnis für: {file_path.name} ---")
            for key, count in per_file_counts.items():
                tqdm.write(f"  Anomalie '{key}': {count:,} betroffene Snapshots gefunden.")
                if per_file_examples[key]:
                    tqdm.write("    Beispiele (snapshot_time, product_id):")
                    for ex in per_file_examples[key]:
                        tqdm.write(f"      - {ex}")
            tqdm.write("-" * 40)

            # Update der globalen Zähler
            for key, value in per_file_counts.items():
                global_anomaly_counts[key] += value

        except Exception as e:
            tqdm.write(f"\nFEHLER beim Verarbeiten der Datei {file_path}: {e}")
            continue

    # --- Finale Zusammenfassung ---
    print("\n" + "=" * 50)
    print("--- FINALES ERGEBNIS DES FORENSISCHEN AUDITS ---")
    print("=" * 50 + "\n")

    anomaly_descriptions = {
        'duplicate_prices_on_side': "1) Gleiche Preise auf unterschiedlichen Levels (pro Seite)",
        'zero_quantity_orders': "3) Snapshots mit Zero-Quantity Orders",
        'sandwiched_zero_quantity': "4) Zero-Quantity Order zwischen Non-Zero Orders",
        'crossed_locked_books': "5) Gekreuzte oder geschlossene Orderbücher (mit >0 Volumen)"
    }

    for key, desc in anomaly_descriptions.items():
        print(f"--- {desc} ---")
        count = global_anomaly_counts.get(key, 0)
        print(f"  Gesamtzahl betroffener Snapshots: {count:,}")

        if examples[key]:
            print(f"  Gespeicherte Beispiele (Datei, Snapshot-Zeit, Produkt-ID):")
            for ex in examples[key]:
                print(f"    - {ex}")
        else:
            print("  Keine Vorkommnisse dieses Typs gefunden.")
        print("-" * 40)

    print("\n--- Audit abgeschlossen ---")


if __name__ == "__main__":
    audit_raw_files(INPUT_DIRECTORY)
