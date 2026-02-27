import os
import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# --- KONFIGURATION ---
# Eingabe-Verzeichnis mit den rohen .parquet-Dateien
RAW_PARQUET_PATH = "../../../data/parquet/00-0-raw/"
# Ausgabe-Verzeichnis für die gebündelten .parquet-Dateien
BATCHED_OUT_PATH = "../../../data/parquet/00-1-product-batches/"
# Pfad zur Protokolldatei für die Fortsetzung
PROGRESS_LOG_FILE = os.path.join("./logs", "progress.log")

# Wie viele Produkte sollen in einer Parquet-Datei gebündelt werden?
PRODUCTS_PER_BATCH = 100

# Spalten, die in den finalen Batches enthalten sein sollen
COLS_TO_LOAD = [
    'snapshot_times',
    'delivery_start',
    'side',
    'price',
    'quantity',
    'orderbook_level',
    "product_id",
    "order_id",
    "creation_time",
]


# ---------------------

def get_processed_files(log_file):
    """Liest die Protokolldatei und gibt ein Set der verarbeiteten Dateien zurück."""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f)


if __name__ == "__main__":
    os.makedirs(BATCHED_OUT_PATH, exist_ok=True)
    all_raw_parquet_files = sorted(glob.glob(os.path.join(RAW_PARQUET_PATH, "*.parquet")))

    if not all_raw_parquet_files:
        print("Keine .parquet-Dateien gefunden.");
        exit()

    # --- logge Konfiguration ---
    print("=" * 50)
    print("--- KONFIGURATION ---")
    print(f"Startzeitpunkt: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Eingabe-Verzeichnis (Rohdaten): {RAW_PARQUET_PATH}")
    print(f"Ausgabe-Verzeichnis (gebündelte Daten): {BATCHED_OUT_PATH}")
    print(f"Protokolldatei für Fortschritt: {PROGRESS_LOG_FILE}")
    print(f"Produkte pro Batch-Datei: {PRODUCTS_PER_BATCH}")
    print(f"Spalten in den Batches: {COLS_TO_LOAD}")
    print("=" * 50 + "\n")

    # --- Schritt 1: Metadaten-Scan ---
    print("Schritt 1: Erstelle Karte von Produkten zu Batches...")
    # (Dieser Teil bleibt unverändert)
    all_unique_delivery_starts = set()
    for parquet_file in tqdm(all_raw_parquet_files, desc="Scanne Produkte"):
        df_meta = pd.read_parquet(parquet_file, columns=['delivery_start'])
        if not df_meta.empty:
            all_unique_delivery_starts.update(pd.to_datetime(df_meta['delivery_start']).unique())
    sorted_products = sorted(list(all_unique_delivery_starts))
    product_to_batch_map = {}
    product_batches = [[p for p in sorted_products[i:i + PRODUCTS_PER_BATCH]] for i in
                       range(0, len(sorted_products), PRODUCTS_PER_BATCH)]
    for batch in product_batches:
        if not batch: continue
        filename = f"{batch[0].strftime('%Y-%m-%dT%H-%M-%S')}_to_{batch[-1].strftime('%Y-%m-%dT%H-%M-%S')}.parquet" if len(
            batch) > 1 else f"{batch[0].strftime('%Y-%m-%dT%H-%M-%S')}.parquet"
        for product_time in batch: product_to_batch_map[product_time] = filename
    print(f"Scan abgeschlossen. {len(product_batches)} logische Batches definiert.")

    # --- Schritt 2: Fortsetzungs-Logik & Sequenzielle Verarbeitung ---
    processed_files = get_processed_files(PROGRESS_LOG_FILE)
    files_to_process = [f for f in all_raw_parquet_files if f not in processed_files]

    if not files_to_process:
        print("\nAlle Dateien bereits verarbeitet. Überspringe zu Schritt 3.")
    else:
        print(
            f"\nSchritt 2: {len(processed_files)} Dateien bereits verarbeitet. Setze Verarbeitung für {len(files_to_process)} Dateien fort...")

    global_stats = defaultdict(int)

    with open(PROGRESS_LOG_FILE, 'a') as log_f:
        pbar = tqdm(files_to_process, desc="Verarbeite Roh-Dateien")
        for raw_file_path in pbar:
            df_raw = pd.read_parquet(raw_file_path, columns=COLS_TO_LOAD)
            if df_raw.empty:
                log_f.write(f"{raw_file_path}\n")  # Auch leere Dateien als verarbeitet markieren
                continue

            rows_read_this_file = len(df_raw)
            global_stats['rows_in'] += rows_read_this_file

            tqdm.write(f"\n--- Verarbeitung von Roh-Datei: {os.path.basename(raw_file_path)} ---")
            tqdm.write(f"  Gelesene Zeilen: {rows_read_this_file:,}")

            df_raw['delivery_start'] = pd.to_datetime(df_raw['delivery_start'])
            df_raw['batch_file'] = df_raw['delivery_start'].map(product_to_batch_map)

            file_duplicates_removed = 0
            file_rows_written = 0

            for batch_filename, data_group in df_raw.groupby('batch_file'):
                if batch_filename is pd.NA or batch_filename is None: continue

                output_path = os.path.join(BATCHED_OUT_PATH, batch_filename)
                data_to_write = data_group.drop(columns=['batch_file'])

                try:
                    rows_before_concat = 0
                    if os.path.exists(output_path):
                        existing_df = pd.read_parquet(output_path)
                        rows_before_concat = len(existing_df)
                        combined_df = pd.concat([existing_df, data_to_write], ignore_index=True)
                    else:
                        combined_df = data_to_write

                    rows_before_dedup = len(combined_df)
                    combined_df.drop_duplicates(inplace=True)
                    rows_after_dedup = len(combined_df)

                    duplicates_in_step = rows_before_dedup - rows_after_dedup
                    net_rows_added = rows_after_dedup - rows_before_concat

                    file_duplicates_removed += duplicates_in_step
                    file_rows_written += net_rows_added

                    combined_df.to_parquet(output_path, index=False)

                except Exception as e:
                    tqdm.write(f"FEHLER beim Anhängen an {batch_filename}: {e}")

            global_stats['duplicates_removed'] += file_duplicates_removed
            global_stats['rows_out_net'] += file_rows_written

            tqdm.write(f"  Entfernte Duplikate in diesem Schritt: {file_duplicates_removed:,}")
            tqdm.write(f"  Netto geschriebene Zeilen in diesem Schritt: {file_rows_written:,}")

            pbar.set_postfix({
                "Gelesen (kum.)": f"{global_stats['rows_in']:,}",
                "Geschrieben (kum.)": f"{global_stats['rows_out_net']:,}",
                "Duplikate (kum.)": f"{global_stats['duplicates_removed']:,}"
            })

            # Markiere die Datei als erfolgreich verarbeitet
            log_f.write(f"{raw_file_path}\n")

    # --- Schritt 3: Finale Deduplizierung ---
    print("\nSchritt 3: Führe finale Deduplizierung pro Batch-Datei durch...")
    final_batch_files = glob.glob(os.path.join(BATCHED_OUT_PATH, "*.parquet"))
    for final_file_path in tqdm(final_batch_files, desc="Dedupliziere Batches"):
        try:
            df_final = pd.read_parquet(final_file_path)
            rows_before = len(df_final)
            df_final.drop_duplicates(inplace=True)
            rows_after = len(df_final)
            if rows_before != rows_after:
                df_final.sort_values(by=['delivery_start', 'snapshot_times'], inplace=True)
                df_final.to_parquet(final_file_path, index=False)
        except Exception as e:
            tqdm.write(f"Fehler beim finalen Deduplizieren von {os.path.basename(final_file_path)}: {e}")

    # --- Finale Zusammenfassung ---
    print("\n" + "=" * 50)
    print("--- FINALE ZUSAMMENFASSUNG DES BATCHING ---")
    print("=" * 50 + "\n")

    # Berechne die finalen Zahlen durch erneutes Einlesen
    total_raw_rows = sum(len(pd.read_parquet(f, columns=[])) for f in all_raw_parquet_files)
    final_total_rows = sum(len(pd.read_parquet(f, columns=[])) for f in final_batch_files)

    print(f"Gesamtzahl der gelesenen Roh-Zeilen (über alle Läufe): {total_raw_rows:,}")
    print(f"Gesamtzahl der Zeilen in den finalen Batch-Dateien: {final_total_rows:,}")

    if total_raw_rows > 0:
        total_dropped = total_raw_rows - final_total_rows
        percent_dropped = (total_dropped / total_raw_rows) * 100 if total_raw_rows > 0 else 0
        print(f"\nGesamte Reduktion (durch Deduplizierung): {total_dropped:,} Zeilen ({percent_dropped:.2f}%)")

    print("\n--- Prozess abgeschlossen ---")
