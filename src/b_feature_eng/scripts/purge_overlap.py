from pathlib import Path

import pandas as pd

import pandas as pd


def purge_overlap(df_prev, df_curr, set_name_prev, set_name_curr,
                  time_col='snapshot_times', product_col='delivery_start'):
    """
    Entfernt Beobachtungen aus df_curr, die zeitlich vor dem Ende von df_prev liegen.
    ZUSÄTZLICH: Entfernt das allererste Produkt (basierend auf delivery_start) aus df_curr,
    um Leakage durch produktübergreifende Features (z.B. Delta zum Vorprodukt) zu verhindern.
    """
    # 1. Sicherstellen, dass die Spalten datetime sind
    df_prev = df_prev.copy()
    df_curr = df_curr.copy()

    df_prev[time_col] = pd.to_datetime(df_prev[time_col])
    df_curr[time_col] = pd.to_datetime(df_curr[time_col])

    if product_col in df_curr.columns:
        df_curr[product_col] = pd.to_datetime(df_curr[product_col])

    # Stats vorher
    rows_orig = len(df_curr)

    print(f"--- Purging: {set_name_prev} -> {set_name_curr} ---")

    # ---------------------------------------------------------
    # SCHRITT A: Zeitliche Überlappung (Snapshot Level) bereinigen
    # ---------------------------------------------------------
    cutoff_time = df_prev[time_col].max()
    print(f"Letzter Snapshot in {set_name_prev}: {cutoff_time}")

    # Filter: Nur Snapshots behalten, die STRIKT nach dem Training-Ende kommen
    df_curr_purged = df_curr[df_curr[time_col] > cutoff_time].copy()

    rows_after_time_purge = len(df_curr_purged)
    dropped_time = rows_orig - rows_after_time_purge

    if df_curr_purged.empty:
        print("WARNUNG: Alle Daten wurden durch den Zeit-Filter entfernt!")
        return df_curr_purged

    # ---------------------------------------------------------
    # SCHRITT B: Erstes Produkt entfernen (Feature Leakage Protection)
    # ---------------------------------------------------------
    # Wir suchen das chronologisch erste Produkt im verbleibenden Set
    first_product_in_curr = df_curr_purged[product_col].min()

    print(f"Erstes Produkt in {set_name_curr} (nach Zeit-Purge): {first_product_in_curr}")
    print(f"-> Entferne dieses Produkt komplett (wegen Rolling/Lag Features)...")

    # Wir behalten nur Produkte, die NACH diesem ersten Produkt starten
    df_curr_final = df_curr_purged[df_curr_purged[product_col] > first_product_in_curr].copy()

    # ---------------------------------------------------------
    # STATISTIKEN
    # ---------------------------------------------------------
    rows_final = len(df_curr_final)
    rows_dropped_product = rows_after_time_purge - rows_final
    total_dropped = rows_orig - rows_final

    print(f"Start von {set_name_curr} (Final):       {df_curr_final[time_col].min()}")
    print(f"Gedroppt (Zeit-Overlap):     {dropped_time}")
    print(f"Gedroppt (Leakage-Produkt):  {rows_dropped_product}")
    print(f"Verlust Gesamt:              {total_dropped} ({total_dropped / rows_orig:.2%})")
    print("-" * 50)

    return df_curr_final


# ---------------------------------------------------------
# ANWENDUNG
# Ersetze 'timestamp' durch den Namen deiner Zeit-Spalte!
# ---------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = Path("/Users/robin/PycharmProjects/Masterarbeit/data/parquet/features/asinh1-reg-clipped/splits")

    TRAIN_PATH = DATA_PATH / "train.parquet"
    VAL_PATH = DATA_PATH / "val_overlap.parquet"
    TEST_PATH = DATA_PATH / "test_overlap.parquet"

    # Lade die Datensätze
    df_train = pd.read_parquet(TRAIN_PATH)
    df_val = pd.read_parquet(VAL_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    # 1. Bereinige Validation Set basierend auf Training Set
    df_val_clean = purge_overlap(
        df_train,
        df_val,
        "Train",
        "Validation",
        time_col='snapshot_times'  # <--- HIER DEINEN SPALTENNAMEN ANPASSEN
    )

    # 2. Bereinige Test Set basierend auf dem (ursprünglichen) Validation Set
    # Wichtig: Wir nehmen df_val (original) als Referenz für das Ende,
    # da das Modell theoretisch bis zum Ende von Val trainiert/validiert wurde.
    df_test_clean = purge_overlap(
        df_val,
        df_test,
        "Validation",
        "Test",
        time_col='snapshot_times'  # <--- HIER DEINEN SPALTENNAMEN ANPASSEN
    )

    # Ergebnisse überprüfen
    print("\nFertig. Neue Größen:")
    print(f"Train: {len(df_train)}")
    print(f"Val:   {len(df_val_clean)}")
    print(f"Test:  {len(df_test_clean)}")

    # Optional: Bereinigte Datensätze speichern

    df_val_clean.to_parquet(DATA_PATH / "val_purged.parquet")
    df_test_clean.to_parquet(DATA_PATH / "test_purged.parquet")
