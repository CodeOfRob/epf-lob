import pandas as pd
from pathlib import Path
from tqdm import tqdm
import collections

# --- KONFIGURATION ---
INPUT_DIRECTORY = Path("../../../data/parquet/03-aggregated-batches")
OUTPUT_DIRECTORY = Path("../../../data/parquet/04-pivoted")


# ---------------------

def pivot_single_file(input_path: Path, output_path: Path) -> collections.defaultdict:
    """
    Lädt eine bereinigte, "lange" Datei, pivotiert sie in ein "breites" Format
    und speichert das Ergebnis. Gibt ein Protokoll der Transformation zurück.
    """
    stats = collections.defaultdict(int)

    df_long = pd.read_parquet(input_path)
    stats['rows_in'] = len(df_long)
    stats['cols_in'] = len(df_long.columns)

    if df_long.empty:
        # Erstelle eine leere Datei, um den Prozess nicht zu unterbrechen
        pd.DataFrame().to_parquet(output_path, index=False)
        stats['rows_out'] = 0
        stats['cols_out'] = 0
        return stats

    # --- Schritt 1: Definiere den Composite Key für die Indizierung ---
    # Diese Spalten identifizieren einen einzigartigen Snapshot eines Produkts
    composite_key = ['snapshot_times', 'delivery_start']

    # --- Schritt 2: Führe die Pivot-Operation durch ---
    # Wir pivotieren NUR die Zustandsvariablen: price und quantity
    df_pivoted = df_long.pivot_table(
        index=composite_key,
        columns=['side', 'orderbook_level'],
        values=['price', 'quantity']
    )

    # --- Schritt 3: Bereinige die Multi-Level-Spaltennamen ---
    # Die Spaltennamen sind jetzt Tupel wie ('price', 'BID', 1).
    # Wir wandeln sie in lesbare Strings wie 'price_1_bid' um.
    df_pivoted.columns = [
        f"{value}_{level}_{side.lower()}"
        for value, side, level in df_pivoted.columns
    ]

    # Der Index wird nach der Pivotierung zu Spalten, also setzen wir ihn zurück
    df_pivoted.reset_index(inplace=True)

    stats['rows_out'] = len(df_pivoted)
    stats['cols_out'] = len(df_pivoted.columns)

    # --- Schritt 4: Speichere die pivotierte Datei ---
    df_pivoted.to_parquet(output_path, index=False)

    return stats


def main():
    """
    Hauptfunktion zur Steuerung des gesamten Pivotierungs-Prozesses.
    """
    if not INPUT_DIRECTORY.is_dir():
        print(f"Fehler: Das Eingabeverzeichnis '{INPUT_DIRECTORY}' wurde nicht gefunden.")
        return

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(list(INPUT_DIRECTORY.glob("*.parquet")))
    if not parquet_files:
        print(f"Fehler: Keine .parquet-Dateien im Verzeichnis '{INPUT_DIRECTORY}' gefunden.")
        return

    # logge Konfiguration
    print("=" * 50)
    print("--- KONFIGURATION ---")
    print(f"Startzeitpunkt: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Eingabe-Verzeichnis (bereinigte Daten): {INPUT_DIRECTORY}")
    print(f"Ausgabe-Verzeichnis (pivotierte Daten): {OUTPUT_DIRECTORY}")
    print("=" * 50 + "\n")

    print(f"Starte Pivotierung von {len(parquet_files)} Parquet-Dateien...")

    # Initialisierung der globalen Aggregatoren
    global_stats = collections.defaultdict(int)

    for file_path in tqdm(parquet_files, desc="Pivotiere Dateien"):
        try:
            output_file_path = OUTPUT_DIRECTORY / file_path.name
            per_file_results = pivot_single_file(file_path, output_file_path)

            # --- Detailliertes Per-File Logging ---
            tqdm.write(f"\n--- Verarbeitung für: {file_path.name} ---")
            tqdm.write(
                f"  Dimensionen (Eingabe): {per_file_results['rows_in']:,} Zeilen, {per_file_results['cols_in']} Spalten")
            tqdm.write(
                f"  Dimensionen (Ausgabe): {per_file_results['rows_out']:,} Zeilen, {per_file_results['cols_out']} Spalten")
            tqdm.write("-" * 30)

            # Update der globalen Statistiken
            for key, value in per_file_results.items():
                if isinstance(value, (int, float)):
                    global_stats[key] += value

        except Exception as e:
            tqdm.write(f"\nFEHLER beim Verarbeiten der Datei {file_path}: {e}")
            continue

    # --- Finale Zusammenfassung ---
    print("\n" + "=" * 50)
    print("--- FINALE ZUSAMMENFASSUNG DER PIVOTIERUNG ---")
    print("=" * 50 + "\n")
    print(f"Gesamtzahl der verarbeiteten Zeilen (Long-Format): {global_stats['rows_in']:,}")
    print(f"Gesamtzahl der erzeugten Snapshot-Zeilen (Wide-Format): {global_stats['rows_out']:,}")

    if 'cols_out' in global_stats:  # Zeige die finale Spaltenanzahl an
        # Wir können nicht einfach aufaddieren, nehmen den Wert aus der letzten Datei
        print(f"Anzahl der Spalten im pivotierten Format: {per_file_results['cols_out']}")

    print("\n--- Pivotierung abgeschlossen ---")


if __name__ == "__main__":
    main()
