import os
import glob
import pyreadr
import pandas as pd
from tqdm import tqdm
import multiprocessing
import warnings

# --- KONFIGURATION ---
# Eingabe-Verzeichnis mit den rohen .rds-Dateien
RDS_FILES_PATH = "../../../data/rds/cache_orderdata_10s/"
# Ausgabe-Verzeichnis für die konvertierten .parquet-Dateien
PARQUET_OUT_PATH = "../../../data/parquet/00-0-raw/"

# Optional: Begrenze die Anzahl der zu verarbeitenden Dateien zum Testen
# Setze auf None, um alle Dateien zu verarbeiten
TOTAL_FILES_TO_PROCESS = None

# Anzahl der parallelen Prozesse (CPUs)
# os.cpu_count() ist eine gute Wahl für maximale Performance
MAX_WORKERS = os.cpu_count() or 4
# ---------------------

# Unterdrücke eine spezifische, harmlose Warnung von pyreadr
warnings.filterwarnings("ignore", message="Could not parse time zone information")


def convert_rds_to_parquet(file_path_tuple):
    """
    Konvertiert eine einzelne .rds-Datei in eine .parquet-Datei.
    Diese Funktion ist so konzipiert, dass sie in einem separaten Prozess läuft.
    """
    rds_path, parquet_path = file_path_tuple

    # Prüfe, ob die Zieldatei bereits existiert
    if os.path.exists(parquet_path):
        return (rds_path, "Skipped - already exists")

    try:
        # Lese die .rds-Datei
        result = pyreadr.read_r(rds_path)
        df = result[None]

        # Schreibe die .parquet-Datei
        df.to_parquet(parquet_path, index=False)

        return (rds_path, "Success")
    except Exception as e:
        # Gebe einen Fehler zurück, der im Hauptprozess protokolliert werden kann
        return (rds_path, f"Error: {e}")


if __name__ == "__main__":
    # Stelle sicher, dass das Ausgabe-Verzeichnis existiert
    os.makedirs(PARQUET_OUT_PATH, exist_ok=True)

    # Finde alle .rds-Dateien
    all_rds_files = sorted(glob.glob(os.path.join(RDS_FILES_PATH, "*.rds")))

    if TOTAL_FILES_TO_PROCESS is not None:
        files_to_process = all_rds_files[:TOTAL_FILES_TO_PROCESS]
    else:
        files_to_process = all_rds_files

    if not files_to_process:
        print(f"Keine .rds-Dateien in '{RDS_FILES_PATH}' gefunden.")
        exit()

    print(f"Bereite die Konvertierung von {len(files_to_process)} .rds-Dateien vor...")

    # Erstelle eine Liste von Aufgaben. Jede Aufgabe ist ein Tupel (Eingabepfad, Ausgabepfad).
    tasks = []
    for rds_file in files_to_process:
        base_name = os.path.basename(rds_file).replace('.rds', '.parquet')
        parquet_file = os.path.join(PARQUET_OUT_PATH, base_name)
        tasks.append((rds_file, parquet_file))

    print(f"Starte parallele Konvertierung mit {MAX_WORKERS} Prozessen...")

    # Erstelle einen Pool von Prozessen
    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        # Führe die Aufgaben parallel aus und zeige den Fortschritt mit tqdm
        # pool.imap_unordered ist speichereffizient für viele Aufgaben
        results = list(
            tqdm(pool.imap_unordered(convert_rds_to_parquet, tasks), total=len(tasks), desc="Konvertiere Dateien"))

    # (Optionale) Auswertung und Fehlerprotokollierung
    success_count = 0
    skipped_count = 0
    error_files = []

    for file, status in results:
        if status == "Success":
            success_count += 1
        elif status == "Skipped - already exists":
            skipped_count += 1
        else:
            error_files.append((file, status))

    print("\n--- Zusammenfassung der Konvertierung ---")
    print(f"Erfolgreich konvertiert: {success_count}")
    print(f"Übersprungen (existiert bereits): {skipped_count}")
    print(f"Fehler aufgetreten: {len(error_files)}")

    if error_files:
        print("\n--- Fehlerdetails ---")
        for file, error_msg in error_files:
            print(f"Datei: {os.path.basename(file)}\n  Fehler: {error_msg}\n")

    print("\nKonvertierung abgeschlossen.")
