import warnings
from pathlib import Path

import joblib


def load_model(model_path):
    """
    Lädt die gespeicherte Pipeline (Preprocessing + Modell)
    und repariert bekannte Versions-Inkompatibilitäten.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

    model = joblib.load(model_path)

    # --- Hotfix für sklearn Versionskonflikte (LogisticRegression) ---
    try:
        # Versuche, den Classifier zu extrahieren
        if hasattr(model, 'named_steps'):
            clf = model.named_steps['classifier']
        else:
            clf = model  # Es ist kein Pipeline-Objekt, sondern direkt der Classifier

        # Check für das spezifische 'multi_class' Problem bei LogReg
        # (Tritt auf, wenn Train-Env != Test-Env)
        if hasattr(clf, 'solver') and not hasattr(clf, 'multi_class'):
            print(f"Fixing missing 'multi_class' attribute for {type(clf).__name__}...")
            clf.multi_class = 'auto'

    except Exception as e:
        warnings.warn(f"Konnte Modell-Patching nicht durchführen: {e}")

    return model


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        start_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype.name

            if col_type not in ['object', 'category', 'datetime64[ns]']:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                    else:
                        # WICHTIG: Wir prüfen NICHT mehr auf float16!
                        # Wir starten direkt bei float32.
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def load_test_data_v2(test_data_path, target_col='label_5min', sample_minutes=None, keep_id_cols=False,
                      min_ttd_minutes=None, max_ttd_minutes=None):
    """
    Lädt die Testdaten und führt ein relatives Downsampling pro Produkt durch.

    Die Zeitstempel werden NICHT verändert (kein floor). Stattdessen wird
    relativ zum Startzeitpunkt jedes Produkts ein Raster gelegt.

    Args:
        test_data_path: Pfad zum Parquet-File.
        target_col: Name der Zielvariable.
        sample_minutes: Integer (z.B. 5). Es wird versucht, exakt alle N Minuten
                        relativ zum Produktstart einen Snapshot zu behalten.
    """
    path = Path(test_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Testdaten nicht gefunden: {path}")

    # Laden
    df = pd.read_parquet(path).pipe(reduce_mem_usage)

    # Sicherstellen, dass snapshot_times datetime ist
    if not pd.api.types.is_datetime64_any_dtype(df['snapshot_times']):
        df['snapshot_times'] = pd.to_datetime(df['snapshot_times'])

    # Sortieren ist essenziell für die Logik "keep='first'"
    df = df.sort_values(['delivery_start', 'snapshot_times'])

    # --- NEU: Relatives Downsampling pro Produkt ---
    if sample_minutes:
        initial_rows = len(df)

        # 1. Startzeitpunkt pro Produkt ermitteln
        # Wir nutzen transform, um den Startzeitpunkt in jede Zeile des Produkts zu schreiben
        df['_product_start'] = df.groupby('delivery_start')['snapshot_times'].transform('min')

        # 2. Zeitdifferenz zum Start berechnen
        delta = df['snapshot_times'] - df['_product_start']

        # 3. Binning berechnen (Ganzzahlige Division durch das Intervall)
        # Beispiel 5min:
        # Minute 0-4.99 -> Bin 0
        # Minute 5-9.99 -> Bin 1
        interval_seconds = sample_minutes * 60
        df['_time_bin'] = (delta.dt.total_seconds() // interval_seconds).astype(int)

        # 4. Duplikate entfernen
        # Wir gruppieren nach Produkt und dem 5-Minuten-Bin und behalten nur den ersten Eintrag.
        # Da wir oben sortiert haben, ist der "erste" Eintrag derjenige, der am nächsten
        # an der theoretischen 5-Minuten-Marke liegt (von links kommend).
        df = df.drop_duplicates(subset=['delivery_start', '_time_bin'], keep='first')

        # Aufräumen
        df = df.drop(columns=['_product_start', '_time_bin'])

        print(f"Downsampling (Relativ {sample_minutes}min): Von {initial_rows:,} auf {len(df):,} Zeilen reduziert.")

    #### MIN / MAX TTD Filterung (optional) ####
    if min_ttd_minutes is not None or max_ttd_minutes is not None:
        print(f"Clipping TTD window to range: {min_ttd_minutes} - {max_ttd_minutes} minutes")
        ttd_minutes = (df["delivery_start"] - df["snapshot_times"]).dt.total_seconds() / 60
        ttd_min_mask = ttd_minutes > min_ttd_minutes if min_ttd_minutes is not None else np.full(len(df), True)
        ttd_max_mask = ttd_minutes < max_ttd_minutes if max_ttd_minutes is not None else np.full(len(df), True)
        print(
            f"Rows to be clipped based on TTD min: {(~ttd_min_mask).sum()} / {len(df)} ({((~ttd_min_mask).sum() / len(df)) * 100:.2f}%)")
        print(
            f"Rows to be clipped based on TTD max: {(~ttd_max_mask).sum()} / {len(df)} ({((~ttd_max_mask).sum() / len(df)) * 100:.2f}%)")
        df = df[ttd_min_mask & ttd_max_mask]
        print(f"Data shape after clipping TTD window: {df.shape}")
    else:
        print("No TTD clipping applied.")

    #### Split X und y ####
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' nicht in Testdaten gefunden.")

    # Trennung X und y
    # ID-Cols logik: Wenn keep_id_cols=True, bleiben sie in X erhalten
    cols_to_drop = [target_col]

    if not keep_id_cols:
        cols_to_drop += ['delivery_start', 'snapshot_times']

    X_test = df.drop(columns=cols_to_drop, errors='ignore')
    y_test = df[target_col]

    # Optional: Metadaten separat zurückgeben, falls keep_id_cols=False war aber du sie brauchst
    # Wenn keep_id_cols=True ist, sind sie eh in X_test
    if not keep_id_cols:
        # Ein kleiner Hack, um die Metadaten trotzdem für Plots verfügbar zu machen,
        # falls du die Funktion so nutzt wie in den vorherigen Schritten.
        # Wir hängen sie einfach wieder an X_test an, wenn der User es nicht explizit verboten hat,
        # ODER du passt den Aufruf an.
        # Sauberer Weg: X_test sollte einfach ein DataFrame sein.
        # Viele ML-Modelle meckern bei Strings, daher ist drop standardmäßig gut.
        # Aber für deine Plot-Funktion brauchst du df_metadata.
        pass

    # Für deine Plotting-Funktion gibst du am besten das df (oder X mit Metadaten) zurück.
    # Da deine Plot-Funktion 'df_metadata' erwartet, ist es oft schlau,
    # X_test MIT den Metadaten zurückzugeben und die Spalten erst direkt vor dem `model.predict()` zu droppen.

    # Hier returnen wir X (features) und y (target).
    # Wenn keep_id_cols=False, fehlen delivery_start/snapshot_times in X_test.
    return X_test, y_test


import pandas as pd
import numpy as np


def align_and_clean_predictions(
        y_true_raw,
        product_keys_raw,
        snapshot_times_raw,
        predictions_dict_raw: dict,
        external_predictions: dict = None
) -> (np.ndarray, pd.Series, pd.Series, dict):
    """
    Synchronisiert und bereinigt einen Satz von Vorhersagen, um sicherzustellen,
    dass alle Modelle auf exakt demselben, lückenlosen Datensatz evaluiert werden.

    Schritte:
    1. Fügt optional externe Vorhersagen (z.B. TabPFN) hinzu.
    2. Erstellt einen zentralen DataFrame.
    3. Entfernt alle Zeilen, in denen mindestens ein Modell eine NaN-Vorhersage hat.
    4. Gibt die bereinigten, perfekt synchronisierten Arrays und das Dictionary zurück.

    Args:
        y_true_raw: Die ursprünglichen, unsynchronisierten wahren Labels.
        product_keys_raw: Die unsynchronisierten Produkt-IDs.
        snapshot_times_raw: Die unsynchronisierten Zeitstempel.
        predictions_dict_raw: Das unsynchronisierte Dictionary mit Modellvorhersagen.
        external_predictions (dict, optional): Dict wie {'ModelName': 'path/to/file.parquet'}.

    Returns:
        Tuple: (y_true_clean, product_keys_clean, snapshot_times_clean, predictions_dict_clean)
    """
    # Lokale Kopie des Dictionaries, um das Original nicht zu verändern
    predictions_clean = predictions_dict_raw.copy()

    # 1. Externe Vorhersagen (z.B. TabPFN) laden und ausrichten
    if external_predictions:
        df_reference = pd.DataFrame({
            'delivery_start': product_keys_raw,
            'snapshot_times': snapshot_times_raw
        })

        for name, path in external_predictions.items():
            print(f"Lade und aligniere externe Vorhersage: '{name}' von {path}")
            try:
                external_df = pd.read_parquet(path)
                # Left-Join, um die Reihenfolge der Referenz zu erzwingen
                df_aligned = pd.merge(
                    df_reference,
                    external_df[['delivery_start', 'snapshot_times', 'y_proba']],
                    on=['delivery_start', 'snapshot_times'],
                    how='left'
                )
                predictions_clean[name] = df_aligned['y_proba'].values
            except Exception as e:
                print(f"  -> Fehler beim Laden von {name}: {e}")
                # Erstelle ein Array mit NaNs, damit es später entfernt wird
                predictions_clean[name] = np.full(len(df_reference), np.nan)

    # 2. Synchronisations-DataFrame erstellen
    df_sync = pd.DataFrame({
        'delivery_start': product_keys_raw,
        'snapshot_times': snapshot_times_raw,
        'y_true': y_true_raw.values if hasattr(y_true_raw, 'values') else y_true_raw
    })
    for name, probas in predictions_clean.items():
        df_sync[name] = probas

    # 3. Alle Zeilen mit mindestens einem NaN entfernen
    n_before = len(df_sync)
    df_sync = df_sync.dropna()
    n_after = len(df_sync)

    if n_before > n_after:
        print(f"⚠️ Bereinigung: {n_before - n_after:,} Zeilen wegen fehlender Vorhersagen entfernt.")
    else:
        print("✅ Alle Vorhersagen sind vollständig und synchronisiert.")

    # 4. Bereinigte Daten extrahieren und zurückgeben
    y_true_final = df_sync['y_true'].values
    product_keys_final = df_sync['delivery_start']
    snapshot_times_final = df_sync['snapshot_times']

    # Dictionary mit den bereinigten Arrays neu aufbauen
    predictions_final = {name: df_sync[name].values for name in predictions_clean.keys()}

    print(f"Finale Anzahl Samples für die Evaluation: {len(y_true_final):,}")

    return y_true_final, product_keys_final, snapshot_times_final, predictions_final
