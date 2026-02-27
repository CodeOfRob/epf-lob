#!/usr/bin/env python
# coding: utf-8

# # config

# In[1]:


import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
from enum import Enum


class SavingStrategy(Enum):
    AGGREGATE = 'aggregate'
    DEDICATED = 'dedicated'


# In[2]:


# Processing Config
ID_COLUMNS = ["snapshot_times", "delivery_start"]
SAVING_STRATEGY: SavingStrategy = SavingStrategy.DEDICATED

# in / out
N_FILES = None  # None for all files
SKIP_IF_FILE_EXISTS = True
LOB_DIR = "/home/sc.uni-leipzig.de/to65jevo/epf-with-ml-on-orderbooks/data/parquet/04-pivoted"
FEATURES_DIR = '/home/sc.uni-leipzig.de/to65jevo/epf-with-ml-on-orderbooks/data/parquet/features/asinh1-reg-clipped'
FEATURES_DIR_SEPARATE = FEATURES_DIR + "/separate/"
FEATURES_DIR_SPLIT = FEATURES_DIR + "/splits/"
FEATURES_DIR_MERGED = FEATURES_DIR + "/merged/"
FEATURES_DIR_SCALER = FEATURES_DIR + "/scaler/"
FEATURES_FILE_MERGED = os.path.join(FEATURES_DIR_MERGED, "all_features_merged.parquet")
FEATURES_FILE_MERGED_CLEANED = os.path.join(FEATURES_DIR_MERGED, "all_features_merged_cleaned.parquet")
generated_files = {}

# split config
TIME_COL = "snapshot_times"
PRODUCT_ID_COL = "delivery_start"

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1
TRAIN_FILE = os.path.join(FEATURES_DIR_SPLIT, "train.parquet")
VAL_FILE = os.path.join(FEATURES_DIR_SPLIT, "val.parquet")
TEST_FILE = os.path.join(FEATURES_DIR_SPLIT, "test.parquet")

# notebook config
PLOTS = False

# scaler config
SCALER_FILE = os.path.join(FEATURES_DIR_SCALER, "scaler.joblib")
TIME_BINS = range(0, 301, 10)  # 5h in 10min steps
SCALER_FEATURE_BLACKLIST_KEYWORDS = [
    "te_",
    "pn_",
    "snapshot_times",
    "delivery_start"
]

# TTD config
MAX_TTD_MINUTES = 300
MIN_TTD_MINUTES = 30


# In[3]:


data = None


# # util

# ## file management

# In[4]:


# function to check whether we should skip calculation based on existing files and flag

def skip_feature(file_name):
    """
    Check if a feature file already exists and whether to skip calculation.
    """
    if SAVING_STRATEGY == SavingStrategy.DEDICATED:
        feature_file_path = os.path.join(FEATURES_DIR_SEPARATE, file_name)
    else:
        feature_file_path = FEATURES_FILE_MERGED

    if SKIP_IF_FILE_EXISTS and os.path.isfile(feature_file_path):
        print(f"Skipping feature calculation, file `{feature_file_path}` already exists.")
        return True
    return False


# In[5]:


# load some amount of files with some specific columns from parquet file

def load_files_with_columns(n_files=N_FILES, columns=None, file_dir=LOB_DIR):
    """
    Load up to `n_files` parquet files from `file_dir`, reading only ID_COLUMNS + columns.
    If `n_files` is None, load all files. Returns a concatenated DataFrame (empty DF if none).
    """
    if columns is None:
        columns = []
    if not os.path.isdir(file_dir):
        raise FileNotFoundError(f"Directory `{file_dir}` does not exist")
    all_files = sorted(os.listdir(file_dir))

    if n_files is not None:
        print("n_files:", n_files)
        all_files = all_files[:n_files]

    dfs = []
    for file in tqdm(all_files, desc="loading parquet files"):
        file_path = os.path.join(file_dir, file)
        try:
            df = pd.read_parquet(file_path, columns=ID_COLUMNS + columns)
        except Exception as e:
            tqdm.write(f"Skipping `{file}`: {e}")
            continue
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=ID_COLUMNS + columns)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df



# In[6]:


# function to load existing all features file with specific columns

def load_existing_features_file(columns=None, file_path=FEATURES_FILE_MERGED):
    """
    Load existing features parquet file from `file_path`, reading only ID_COLUMNS + columns.
    If the file does not exist, returns an empty DataFrame with the specified columns.
    """
    if columns is None:
        columns = []
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path, columns=ID_COLUMNS + columns)
        return df
    else:
        print(f"Features file `{file_path}` does not exist. Returning empty DataFrame.")
        return pd.DataFrame(columns=ID_COLUMNS + columns)


# In[7]:


# function to handle saving globally and adjust saving strategy

def save_features(df, file_name=None, file_dir=FEATURES_DIR_SEPARATE, file_path=FEATURES_FILE_MERGED):
    """
    Save features from `df` using the specified `strategy`.
    """
    print("Saving features with strategy:", SAVING_STRATEGY)
    if SAVING_STRATEGY == SavingStrategy.AGGREGATE:
        save_to_aggregate_features_file(df, file_path)
    elif SAVING_STRATEGY == SavingStrategy.DEDICATED:
        if file_name is None:
            raise ValueError("file_name must be provided for DEDICATED saving strategy")
        save_to_dedicated_features_file(df, file_name, file_dir)
    else:
        raise ValueError(f"Unknown saving strategy: {SAVING_STRATEGY}")


# In[8]:


# function to save new columns to existing features file

def save_to_aggregate_features_file(df, file_path=FEATURES_FILE_MERGED):
    """
    Save new features from `df` to the existing features parquet file at `file_path`.
    Only new columns (not in existing file) will be added.
    """
    if os.path.isfile(file_path):
        existing_df = pd.read_parquet(file_path)
        new_columns = [col for col in df.columns if col not in existing_df.columns]
        if new_columns:
            updated_df = pd.concat([existing_df, df[new_columns]], axis=1)
            updated_df.to_parquet(file_path, index=False)
            print(f"Added new columns: {new_columns}")
        else:
            print("No new columns to add.")
    else:
        df.to_parquet(file_path, index=False)
        print(f"Created new features file with columns: {df.columns.tolist()}")

    generated_files["aggregate"] = file_path


# In[9]:


# function to save new columns to dedicated feature file in features directory

def save_to_dedicated_features_file(df, file_name, features_dir=FEATURES_DIR_SEPARATE):
    """
    Save new features from `df` to a dedicated parquet file in `features_dir`.
    The file will be named based on the new feature columns.
    """
    if not os.path.isdir(features_dir):
        os.makedirs(features_dir)

    feature_columns = [col for col in df.columns if col not in ID_COLUMNS]
    if not feature_columns:
        print("No new features to save.")
        return

    feature_file_path = os.path.join(features_dir, file_name)

    df.to_parquet(feature_file_path, index=False)

    generated_files[file_name] = feature_file_path

    print(f"Saved new features to `{feature_file_path}` with columns: {feature_columns}")


# ## calculations

# In[10]:


def calculate_returns(df, value_col, horizons_min, direction='past', id_col='delivery_start',
                      time_col='snapshot_times'):
    """
    Berechnet Returns (Differenzen) in die Vergangenheit ODER Zukunft.

    Args:
        direction (str): 'past' (für Features: t - k) oder 'future' (für Labels: t + k).
    """
    # 1. Sortieren & Typisieren (wie gehabt)
    df = df.sort_values(by=[time_col]).reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    lookup_df = df[[id_col, time_col, value_col]].copy()
    lookup_df.columns = [id_col, 'lookup_timestamp', 'lookup_value']

    for horizon in tqdm(horizons_min, desc=f"Calc {direction} returns for {value_col}"):
        target_time_col = f'target_time_{horizon}m'

        # 2. Zielzeit & Suchrichtung bestimmen
        if direction == 'past':
            df[target_time_col] = df[time_col] - pd.Timedelta(minutes=horizon)
            merge_dir = 'backward'  # Suche <= Zielzeit
        else:  # future
            df[target_time_col] = df[time_col] + pd.Timedelta(minutes=horizon)
            merge_dir = 'forward'  # Suche >= Zielzeit

        # 3. Merge As-Of
        merged = pd.merge_asof(
            left=df,
            right=lookup_df,
            left_on=target_time_col,
            right_on='lookup_timestamp',
            by=id_col,
            direction=merge_dir
        )

        # 4. Validierung
        if direction == 'past':
            # Der gefundene Wert muss in der Vergangenheit liegen (kleiner als Jetzt)
            valid_mask = merged['lookup_timestamp'] < df[time_col]
            # Return: Aktuell - Vergangenheit
            diff = df[value_col] - merged['lookup_value']
        else:
            # Der gefundene Wert muss in der Zukunft liegen (größer als Jetzt)
            valid_mask = merged['lookup_timestamp'] > df[time_col]
            # Return: Zukunft - Aktuell (Standard für Labels)
            diff = merged['lookup_value'] - df[value_col]

        # 5. Zuweisen
        suffix = 'prev' if direction == 'past' else 'next'
        new_col_name = f'{value_col}_return_{suffix}_{horizon}min'

        df[new_col_name] = np.where(valid_mask, diff, np.nan)
        df.drop(columns=[target_time_col], inplace=True)

    return df


# In[11]:


# function to lag a column by seconds per product

def create_time_based_lags(df, target_col, lags_seconds, id_col='delivery_start', time_col='snapshot_times'):
    """
    Erstellt Lags für eine spezifische Spalte basierend auf Zeit-Intervallen.

    Logik:
    1. Berechnet Ziel-Zeitpunkt: t_ziel = t_aktuell - lag_seconds
    2. Sucht den letzten verfügbaren Wert vor oder genau zu diesem Zeitpunkt (merge_asof backward).

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name der Spalte, die gelagged werden soll (z.B. 'mid_price').
        lags_seconds (list): Liste von Integers (Sekunden), z.B. [10, 30, 60].
        id_col (str): Spalte zur Gruppierung.
        time_col (str): Zeitstempel-Spalte.

    Returns:
        pd.DataFrame: DataFrame mit den neuen Lag-Spalten.
    """
    # 1. Sortieren und Index resetten (Essentiell für merge_asof und Zuweisung!)
    df = df.sort_values(by=[time_col]).reset_index(drop=True)

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Lookup-Tabelle (rechte Seite)
    lookup_df = df[[id_col, time_col, target_col]].copy()
    lookup_df.columns = [id_col, 'past_timestamp', 'lagged_value']

    for lag_sec in tqdm(lags_seconds, desc=f"Creating time-based lags for {target_col}"):
        # 2. Berechne Ziel-Zeitpunkt
        target_time_col = f'target_time_lag_{lag_sec}s'
        df[target_time_col] = df[time_col] - pd.Timedelta(seconds=lag_sec)

        # 3. Merge As-Of (Backward)
        # Findet den letzten Wert, dessen Zeitstempel <= target_time ist
        merged = pd.merge_asof(
            left=df,
            right=lookup_df,
            left_on=target_time_col,
            right_on='past_timestamp',
            by=id_col,
            direction='backward'
        )

        # 4. Spaltenname generieren
        new_col_name = f'{target_col}_lag_{lag_sec}s'

        # Zuweisen (Dank reset_index passt die Reihenfolge)
        # Beachte: merge_asof backward gibt automatisch NaN zurück, wenn kein Wert in der
        # Vergangenheit gefunden wird (z.B. ganz am Anfang der Zeitreihe).
        # Wir müssen hier keinen extra Sicherheits-Check machen wie beim 'forward',
        # da 'backward' per Definition nicht in die Zukunft schauen kann.
        df[new_col_name] = merged['lagged_value']

        # Cleanup
        df.drop(columns=[target_time_col], inplace=True)

    return df


# In[12]:


# function to calculate rolling mean by n seconds per product
def create_time_based_rolling_means(df, target_col, windows_seconds, id_col='delivery_start',
                                    time_col='snapshot_times'):
    """
    Berechnet gleitende Durchschnitte. Robust gegen duplizierte Zeitstempel über Produkte hinweg.
    """
    # 1. Sicherstellen, dass der Zeitstempel datetime ist
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # 2. Sortieren ist wichtig für rolling, aber wir behalten den originalen Index bei!
    # Wir sortieren und merken uns die Reihenfolge.
    df = df.sort_values(by=[id_col, time_col])

    # 3. Temporärer DataFrame mit Zeit-Index für die Berechnung
    # Wir setzen den Index auf (id_col, time_col), um Eindeutigkeit zu schaffen (hoffentlich)
    # Aber noch besser: Wir nutzen die `on`-Option von rolling() in neueren Pandas Versionen
    # ODER wir setzen den Index temporär, rechnen und setzen ihn zurück.

    # Der sicherste Weg, der immer funktioniert:
    # Wir setzen den Zeitstempel als Index, führen die Operation aus, und nutzen den originalen Index zum Mergen/Zuweisen.

    df_temp = df.copy()
    df_temp = df_temp.set_index(time_col)

    for window_sec in tqdm(windows_seconds, f"Creating time-based rolling means for {target_col}"):
        window_str = f'{window_sec}s'
        new_col_name = f'{target_col}_MA_{window_sec}s'

        # Berechnung
        # Das Ergebnis von rolling() hat den gleichen Index wie der Input (hier: Zeitstempel).
        # Da Zeitstempel nicht eindeutig sind (mehrere Produkte zur gleichen Zeit),
        # müssen wir aufpassen.

        # TRICK: Wir nutzen den Gruppen-Ansatz, aber sorgen dafür, dass wir das Ergebnis
        # direkt als Array oder Serie zuweisen können, die zur sortierten 'df' passt.

        calculated_series = df_temp.groupby(id_col)[target_col] \
            .rolling(window=window_str, min_periods=int((window_sec / 10) * 0.5)) \
            .mean()

        # calculated_series hat jetzt einen MultiIndex (product_id, snapshot_times).
        # Wir müssen diesen zurück in die Form von 'df' bringen.

        # Da 'df' bereits nach [id_col, time_col] sortiert ist, sollte die Reihenfolge der Werte
        # in 'calculated_series' exakt der Reihenfolge der Zeilen in 'df' entsprechen!
        # Wir können also einfach die Werte (.values) zuweisen.

        df[new_col_name] = calculated_series.values

    return df


# In[13]:


# function to calculate rolling std by n seconds per product (realized volatility)


def create_time_based_realized_volatility(df, target_col, windows_seconds, id_col='delivery_start',
                                          time_col='snapshot_times'):
    """
    Berechnet die Realized Volatility (Rolling Std) über definierte Zeitfenster.
    Nutzt den robusten 'Sort & Assign by Value' Ansatz.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Spalte, für die die Vola berechnet wird (idealerweise Returns).
        windows_seconds (list): Liste von Integers (Sekunden).

    Returns:
        pd.DataFrame: Der DataFrame mit den neuen RV-Spalten.
    """
    # 1. Sicherstellen, dass der Zeitstempel datetime ist
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # 2. Sortieren nach Produkt und Zeit (Essentiell für korrekte Zuordnung!)
    df = df.sort_values(by=[id_col, time_col])

    # 3. Temporärer DataFrame mit Zeit-Index für die Rolling-Logik
    df_temp = df.copy()
    df_temp = df_temp.set_index(time_col)

    for window_sec in tqdm(windows_seconds, desc=f"Creating time-based realized volatility for {target_col}"):
        window_str = f'{window_sec}s'
        # Naming Convention: Wenn target 'return' ist, heißt es oft nur RV_...
        # Hier generisch: target_RV_window
        new_col_name = f'{target_col}_RV_{window_sec}s'

        # Berechnung
        # min_periods=2: StdDev braucht mind. 2 Punkte.
        calculated_series = df_temp.groupby(id_col)[target_col] \
            .rolling(window=window_str, min_periods=int((window_sec / 10) * 0.5)) \
            .std()

        # Zuweisung der Werte (direktes Array, da Sortierung identisch)
        df[new_col_name] = calculated_series.values

    return df


# In[14]:


# function to sin / cos encode a cyclical feature
def encode_cyclical_feature(df, col_name, period, delete_original_col=True):
    """
    Encode a cyclical feature using sine and cosine transformations.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Name of the cyclical column to encode.
        period (int): The period of the cycle (e.g., 24 for hours in a day).
        delete_original_col (bool): Delete original column if it exists.
    Returns:
        pd.DataFrame: DataFrame with new sine and cosine encoded columns.
    """
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / period)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / period)

    if delete_original_col:
        df.drop(columns=[col_name], inplace=True)
    return df


# In[15]:


# function for calculating mid price
def calculate_mid_price(df):
    """
    Calculate mid price and add it as a new column 'mid_price' to the DataFrame.
    Assumes columns 'price_1_bid' and 'price_1_ask' exist in the DataFrame.
    """
    if 'price_1_bid' not in df.columns or 'price_1_ask' not in df.columns:
        raise ValueError("DataFrame must contain 'best_bid_price' and 'best_ask_price' columns")
    df['mid_price'] = (df['price_1_bid'] + df['price_1_ask']) / 2
    return df


# In[16]:


# function for calculating weighted mid price
def calculate_weighted_mid_price(df):
    """
    Calculate weighted mid price as volume-weighted average of bid and ask prices.
    Adds a new column 'weighted_mid_price' to the DataFrame.
    Assumes columns 'price_1_bid', 'price_1_ask', 'quantity_1_bid', and 'quantity_1_ask' exist in the DataFrame.
    If quantity of a single side is zero, the mid price defaults to the other side's price.
    """
    required_cols = ['price_1_bid', 'price_1_ask', 'quantity_1_bid', 'quantity_1_ask']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    bid_contrib = df['price_1_bid'] * df['quantity_1_bid']
    ask_contrib = df['price_1_ask'] * df['quantity_1_ask']
    total_quantity = df['quantity_1_bid'] + df['quantity_1_ask']

    # Avoid division by zero
    df['weighted_mid_price'] = np.where(
        total_quantity > 0,
        (bid_contrib + ask_contrib) / total_quantity,
        np.where(
            df['quantity_1_bid'] > 0,
            df['price_1_bid'],
            np.where(
                df['quantity_1_ask'] > 0,
                df['price_1_ask'],
                np.nan  # both quantities are zero
            )
        )
    )
    return df


# In[17]:


# function to calculate bid-ask spread
def calculate_bid_ask_spread(df):
    """
    Calculate bid-ask spread and add it as a new column 'bid_ask_spread' to the DataFrame.
    Assumes columns 'best_bid_price' and 'best_ask_price' exist in the DataFrame.
    If either price is missing, spread is set to 0.
    """
    if 'price_1_bid' not in df.columns or 'price_1_ask' not in df.columns:
        raise ValueError("DataFrame must contain 'best_bid_price' and 'best_ask_price' columns")
    df['bid_ask_spread'] = df['price_1_ask'] - df['price_1_bid']
    df['bid_ask_spread'] = df['bid_ask_spread'].fillna(0)
    return df


# In[18]:


# function for calculating difference in a column for next / prev delivery hour in same snapshot
def add_cross_product_diff_robust(df, target_col, neighbor_offset_hours=1, tolerance_seconds=60):
    """
    Berechnet die Differenz zum Nachbarprodukt.
    Nutzt 'merge_asof', um auch dann einen Wert zu finden, wenn die Zeitstempel
    nicht exakt übereinstimmen (sondern nur nah beieinander liegen).
    """
    # WICHTIG: merge_asof braucht sortierte Daten!
    df = df.sort_values('snapshot_times')

    join_key_col = 'target_neighbor_delivery'
    df[join_key_col] = df['delivery_start'] + pd.Timedelta(hours=neighbor_offset_hours)

    lookup = df[['delivery_start', 'snapshot_times', target_col]].copy()
    lookup = lookup.sort_values('snapshot_times')  # Lookup muss auch sortiert sein
    lookup.columns = [join_key_col, 'snapshot_times', 'neighbor_value']

    # Merge As-Of
    # "Finde für jede Zeile in df den Eintrag in lookup, dessen Zeitstempel
    #  kleiner oder gleich ist (direction='backward'), aber maximal 60 Sekunden alt."
    # by=[...] stellt sicher, dass wir nur innerhalb des richtigen Nachbar-Produkts suchen.
    merged = pd.merge_asof(
        df,
        lookup,
        on='snapshot_times',
        by=join_key_col,
        direction='backward',
        tolerance=pd.Timedelta(seconds=tolerance_seconds)
    )

    new_col_name = f'diff_{target_col}_neighbor_{neighbor_offset_hours}h'
    merged[new_col_name] = merged[target_col] - merged['neighbor_value']

    # Aufräumen (Sortierung wiederherstellen ist hier implizit, da wir am Anfang sortiert haben)
    merged = merged.drop(columns=[join_key_col, 'neighbor_value'])

    return merged


# In[19]:


# function for creation of flags indicating whether to the time of the snapshot the neighboring products are actively traded

def create_active_flags_robust(df, time_grid='10s'):
    """
    Berechnet Flags basierend auf einem normalisierten Zeitraster.
    Vermeidet das Flackern von asynchronen Ticks.
    """
    df = df.copy()

    # 1. Zeitraster normalisieren (Flooring)
    # Das "fängt" alle Ticks, die in diesem 10s-Fenster liegen.
    df['time_bin'] = df['snapshot_times'].dt.floor(time_grid)

    # 2. Aktivitäts-Tabelle erstellen (Welches Produkt war in welchem Bin aktiv?)
    # Wir nehmen nur die eindeutigen Paare (Produkt, Zeitbin)
    # drop_duplicates ist extrem schnell.
    active_grid = df[['delivery_start', 'time_bin']].drop_duplicates()
    active_grid['is_active'] = 1

    # 3. Join-Logik für Nachbarn
    # Wir joinen das Grid gegen sich selbst.

    # Für Prev (-1h): Wir suchen im Grid nach (Delivery-1h, TimeBin)
    # Wir berechnen temporär das Ziel
    active_grid['target_prev'] = active_grid['delivery_start'] - pd.Timedelta(hours=1)

    # Join: Finden wir einen Eintrag für (target_prev, time_bin)?
    # Wir joinen active_grid (links) mit active_grid (rechts)
    merged_prev = pd.merge(
        active_grid,
        active_grid[['delivery_start', 'time_bin', 'is_active']],  # Rechte Seite (Lookup)
        left_on=['target_prev', 'time_bin'],
        right_on=['delivery_start', 'time_bin'],
        how='left',
        suffixes=('', '_match')
    )

    # Für Next (+1h):
    active_grid['target_next'] = active_grid['delivery_start'] + pd.Timedelta(hours=1)

    merged_next = pd.merge(
        active_grid,
        active_grid[['delivery_start', 'time_bin', 'is_active']],
        left_on=['target_next', 'time_bin'],
        right_on=['delivery_start', 'time_bin'],
        how='left',
        suffixes=('', '_match')
    )

    # 4. Ergebnisse zurückmappen auf den Original-DataFrame
    # Wir haben jetzt für jedes (Produkt, Zeitbin) die Flags.
    # Wir müssen das zurück an die rohen Ticks joinen.

    # Bereite die Flags vor
    active_grid['is_prev_product_active'] = merged_prev['is_active_match'].fillna(0).astype(int)
    active_grid['is_next_product_active'] = merged_next['is_active_match'].fillna(0).astype(int)

    # Finaler Join an die Originaldaten
    # Wir joinen über (delivery_start, time_bin)
    final_df = pd.merge(
        df,
        active_grid[['delivery_start', 'time_bin', 'is_prev_product_active', 'is_next_product_active']],
        on=['delivery_start', 'time_bin'],
        how='left'
    )

    # Aufräumen
    final_df = final_df.drop(columns=['time_bin'])

    return final_df


# In[20]:


# function for calculating difference in a column for next / prev delivery hour in same snapshot robustly

def create_spillover_diffs_robust(df, target_cols, neighbor_offset_hours=1, tolerance_seconds=60):
    """
    Berechnet die Differenz zum Nachbarn.
    Iteriert über eindeutige 'delivery_start' Werte, um saubere Time-Series-Joins zu garantieren.
    """
    df = df.sort_values('snapshot_times').copy()

    # Wir brauchen eine Liste aller Produkte, um Daten schnell zu finden
    # (Ein GroupBy-Objekt ist hier effizient)
    grouped = df.groupby('delivery_start')

    # Ergebnis-Listen
    results = []

    # Iteriere über jedes Produkt im aktuellen Batch
    for delivery, group in grouped:
        # 1. Bestimme den Nachbarn
        neighbor_delivery = delivery + pd.Timedelta(hours=neighbor_offset_hours)

        # 2. Hole die Daten des Nachbarn (falls im Batch vorhanden)
        # Achtung: Wenn der Nachbar in einem ANDEREN File liegt, finden wir ihn hier nicht.
        # Aber du hast gesagt, du hast "Rolling Window" Loading.
        # Falls der Nachbar fehlt, ist das Ergebnis NaN (und wird später 0).

        neighbor_data = None
        if neighbor_delivery in grouped.groups:
            neighbor_data = grouped.get_group(neighbor_delivery)[['snapshot_times'] + target_cols].sort_values(
                'snapshot_times')

        # 3. Wenn Nachbar nicht da -> Alles 0 (bzw. NaN und dann fillna)
        if neighbor_data is None or neighbor_data.empty:
            for col in target_cols:
                group[f'{col}_diff_{neighbor_offset_hours}h'] = 0.0
            results.append(group)
            continue

        # 4. Wenn Nachbar da -> Merge AsOf
        # Da wir jetzt nur ZWEI saubere Zeitreihen haben, funktioniert merge_asof perfekt!
        merged = pd.merge_asof(
            group,
            neighbor_data,
            on='snapshot_times',
            direction='backward',
            tolerance=pd.Timedelta(seconds=tolerance_seconds),
            suffixes=('', '_neighbor')
        )

        # 5. Differenzen berechnen
        for col in target_cols:
            # Berechne Diff
            diff = merged[col] - merged[f'{col}_neighbor']
            # Imputiere 0.0 wo kein Match gefunden wurde (Toleranz überschritten oder Lücke)
            group[f'{col}_diff_{neighbor_offset_hours}h'] = diff.fillna(0.0).values

        results.append(group)

    # Wieder zusammenfügen
    return pd.concat(results).sort_index()


def create_spillover_diffs_robust_ttd(df, target_cols, neighbor_offset_hours=1, tolerance_minutes=1.0):
    """
    Berechnet die Differenz zum Nachbarprodukt basierend auf der RELATIVEN ZEIT (time_to_delivery).
    Vergleicht Anomalie-Scores zum gleichen Zeitpunkt im Lebenszyklus.
    """
    df = df.copy()

    # 1. Time-to-Delivery berechnen (falls nicht da)
    if 'time_to_delivery_min' not in df.columns:
        df['time_to_delivery_min'] = (df['delivery_start'] - df['snapshot_times']).dt.total_seconds() / 60.0

    # Sortieren nach TTD ist wichtig für merge_asof
    # Achtung: TTD ist absteigend (180 -> 0). merge_asof braucht aufsteigend.
    # Wir sortieren also nach -TTD oder einfach aufsteigend, aber müssen aufpassen.
    # Am einfachsten: Wir sortieren nach TTD aufsteigend (0 -> 180).
    # Das bedeutet, die Snapshots sind "umgekehrt chronologisch".
    df = df.sort_values('time_to_delivery_min')

    grouped = df.groupby('delivery_start')
    results = []

    for delivery, group in grouped:
        # 1. Bestimme den Nachbarn (z.B. +1h -> Nächste Lieferung)
        # Wenn wir Prev (-1) analysieren wollen: Wir vergleichen UNS (t) mit dem VORGÄNGER (t-1).
        neighbor_delivery = delivery + pd.Timedelta(hours=neighbor_offset_hours)

        neighbor_data = None
        if neighbor_delivery in grouped.groups:
            # Hole Nachbar und sortiere auch nach TTD
            neighbor_data = grouped.get_group(neighbor_delivery)[['time_to_delivery_min'] + target_cols].sort_values(
                'time_to_delivery_min')

        if neighbor_data is None or neighbor_data.empty:
            for col in target_cols:
                group[f'{col}_diff_{neighbor_offset_hours}h'] = np.nan
            results.append(group)
            continue

        # 2. Merge AsOf auf time_to_delivery_min
        # direction='nearest' ist hier am besten, da wir den ähnlichsten Punkt im Zyklus wollen.
        merged = pd.merge_asof(
            group,
            neighbor_data,
            on='time_to_delivery_min',
            direction='nearest',
            tolerance=tolerance_minutes,  # z.B. +/- 1 Minute Abweichung erlaubt
            suffixes=('', '_neighbor')
        )

        # 3. Differenzen berechnen
        for col in target_cols:
            # Diff der Anomalie-Scores
            diff = merged[col] - merged[f'{col}_neighbor']
            group[f'{col}_diff_{neighbor_offset_hours}h'] = diff.fillna(0.0).values

        results.append(group)

    # Wiederherstellung der ursprünglichen Sortierung (nach Zeit)
    return pd.concat(results).sort_values(['delivery_start', 'snapshot_times'])


# ## visualizations

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize


def analyze_feature_compact(df, feature_col, winsor_limits=(0.01, 0.01)):
    """
    Erstellt eine kompakte Analyse (1 Zeile, 2 Plots) für ein Feature:
    1. Winsorisierter KDE Plot (Verteilung)
    2. Lifecycle-Analyse (Verlauf über Time-to-Delivery mit Quartilsband)
       -> Berechnet Time-to-Delivery automatisch aus delivery_start und snapshot_times.
    """

    if PLOTS is False:
        print("PLOTS disabled in config.")
        return

    # Setup der Figur
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(f"Feature Analyse: {feature_col}", fontsize=16, y=1.05)

    # Check: Existiert das Feature?
    if feature_col not in df.columns:
        print(f"Fehler: Spalte {feature_col} nicht gefunden.")
        return

    # --- 1. Winsorisierter KDE Plot ---
    data = df[feature_col].dropna().values

    if len(data) > 0:
        # Winsorisieren (Extreme Outlier kappen für den Plot)
        data_winsor = winsorize(data, limits=winsor_limits)

        sns.histplot(data_winsor, kde=True, ax=axes[0], color='skyblue', edgecolor='white', stat='density')
        axes[0].set_title(f"Verteilung (Winsorized 1% - 99%)")
        axes[0].set_xlabel(feature_col)

        # Stats
        mean_val = np.mean(data)
        median_val = np.median(data)
        axes[0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        axes[0].axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.2f}')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Keine Daten", ha='center', va='center')

    # --- 2. Lifecycle Analyse ---

    # Wir brauchen die Zeitspalten
    required_cols = ['delivery_start', 'snapshot_times', feature_col]
    if all(col in df.columns for col in required_cols):
        # Arbeite auf einer Kopie mit den nötigen Spalten
        df_life = df[required_cols].dropna().copy()

        # Typisierung sicherstellen
        df_life['delivery_start'] = pd.to_datetime(df_life['delivery_start'])
        df_life['snapshot_times'] = pd.to_datetime(df_life['snapshot_times'])

        # Berechne Time to Delivery in Minuten
        df_life['ttd_min'] = (df_life['delivery_start'] - df_life['snapshot_times']).dt.total_seconds() / 60

        # Erstelle Bins (Wir nehmen 5-Minuten-Schritte für hohe Auflösung)
        # Bereich: 0 bis Max (z.B. 300 Min)
        max_min = df_life['ttd_min'].max()
        bins = np.arange(0, max_min + 5, 5)  # Alle 5 Minuten

        df_life['time_bin'] = pd.cut(df_life['ttd_min'], bins=bins)

        # Aggregation
        grp = df_life.groupby('time_bin')[feature_col].agg(
            ['mean', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        grp.columns = ['mean', 'median', 'q25', 'q75']

        # X-Achse: Mitte des Bins
        grp['x'] = [i.mid for i in grp.index]

        # Plot
        axes[1].plot(grp['x'], grp['mean'], color='blue', label='Mean', linewidth=2)
        axes[1].plot(grp['x'], grp['median'], color='darkblue', linestyle='--', label='Median')
        axes[1].fill_between(grp['x'], grp['q25'], grp['q75'], color='blue', alpha=0.15, label='IQR (25-75%)')

        axes[1].set_title("Verlauf über Produkt-Lebenszyklus")
        axes[1].set_xlabel("Minuten bis Lieferung")
        axes[1].set_ylabel(feature_col)
        axes[1].invert_xaxis()  # Countdown-Style: 180 -> 0
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    else:
        axes[1].text(0.5, 0.5, "Zeitspalten fehlen", ha='center', va='center')

    plt.tight_layout()
    plt.show()


# # features

# ## mid price returns

# In[22]:


# calculate mid price returns
# constraints:
# - as we work with energy price data, price can be zero or negative, so we cannot use log returns --> use absolute differences instead
# - all calculations should be done per product using the product identifier "delivery_start"
# in order to capture multiple different moments, we deploy multiple horizons the current mid price to be subtracted from

FEATURE_NAME = "mid_price_returns"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    ID_COL = "delivery_start"
    TIME_COL = "snapshot_times"
    COLS_TO_READ = ['price_1_ask', 'price_1_bid']
    HORIZONS_MIN = [1, 5, 15, 30]

    print("### Calculating mid price returns... ###")

    # load necessary columns
    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # calculate mid price
    data = calculate_mid_price(data)
    print("Mid price calculated.")

    # calculate mid price returns
    data = calculate_returns(data, value_col='mid_price', horizons_min=HORIZONS_MIN, id_col=ID_COL, time_col=TIME_COL)
    print("Past mid price differences calculated.")

    # Save only the new feature columns along with ID columns
    save_features(data[ID_COLUMNS + [f'mid_price_return_prev_{h}min' for h in HORIZONS_MIN]], f"{FEATURE_NAME}.parquet")
    print("Mid price return features saved in" + f" {FEATURE_NAME}.parquet")


# In[23]:


analyze_feature_compact(data, 'mid_price_return_prev_1min')
analyze_feature_compact(data, 'mid_price_return_prev_1min_absolute_anomaly')


# ## weighted mid price returns

# In[24]:


# calculate weighted mid price returns
# similar constraints as mid price returns

FEATURE_NAME = "weighted_mid_price_returns"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    ID_COL = "delivery_start"
    TIME_COL = "snapshot_times"
    COLS_TO_READ = ['price_1_ask', 'price_1_bid', 'quantity_1_ask', 'quantity_1_bid']
    HORIZONS_MIN = [1, 5, 15, 30]

    print("### Calculating weighted mid price returns... ###")

    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # start by calculating mid price
    data = calculate_weighted_mid_price(data)
    print("Weighted mid price calculated.")

    # calculate weighted mid price returns
    data = calculate_returns(data, value_col='weighted_mid_price', horizons_min=HORIZONS_MIN, id_col=ID_COL,
                             time_col=TIME_COL)
    print("Past weighted mid price differences calculated.")

    # Save only the new feature columns along with ID columns
    save_features(data[ID_COLUMNS + [f'weighted_mid_price_return_prev_{h}min' for h in HORIZONS_MIN]],
                  f"{FEATURE_NAME}.parquet")


# In[25]:


# analyze weighted mid price return feature
analyze_feature_compact(data, 'weighted_mid_price_return_5min')
analyze_feature_compact(data, 'weighted_mid_price_return_5min_as')


# ## mp wmp difference

# In[26]:


# calculate differences between mid price and weighted mid price over the generated horizons


FEATURE_NAME = "mp_wmp_return_differences"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    ID_COL = "delivery_start"
    TIME_COL = "snapshot_times"
    COLS_TO_READ = ['price_1_ask', 'price_1_bid', 'quantity_1_ask', 'quantity_1_bid']
    HORIZONS_MIN = [1, 5, 15, 30]

    print("### Calculating MP-WMP return differences... ###")

    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # start by calculating mid price and weighted mid price
    data = calculate_mid_price(data)
    data = calculate_weighted_mid_price(data)
    print("Mid price and Weighted mid price calculated.")

    # calculate returns for both prices
    data = calculate_returns(data, value_col='mid_price', horizons_min=HORIZONS_MIN, id_col=ID_COL, time_col=TIME_COL)
    data = calculate_returns(data, value_col='weighted_mid_price', horizons_min=HORIZONS_MIN, id_col=ID_COL,
                             time_col=TIME_COL)
    print("Past mid price and weighted mid price differences calculated.")

    # calculate differences between mid price returns and weighted mid price returns
    for horizon in HORIZONS_MIN:
        mid_price_col = f'mid_price_return_prev_{horizon}min'
        wmp_col = f'weighted_mid_price_return_prev_{horizon}min'
        diff_col = f'mp_wmp_return_diff_prev_{horizon}min'
        data[diff_col] = data[mid_price_col] - data[wmp_col]
        print(f"Calculated difference for horizon {horizon}min.")

    # Save only the new feature columns along with ID columns
    save_features(data[ID_COLUMNS + [f'mp_wmp_return_diff_prev_{h}min' for h in HORIZONS_MIN]],
                  f"{FEATURE_NAME}.parquet")


# In[27]:


analyze_feature_compact(data, 'mp_wmp_return_diff_5min')
analyze_feature_compact(data, "mp_wmp_return_diff_5min_as")


# ## realized volatility of mid price

# In[28]:


# calculate realized volatility of mid price returns
# constraints:
# - realized volatility is defined as rolling standard deviation of mid price returns
# - rolling windows are defined in seconds

FEATURE_NAME = "mid_price_realized_volatility"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    COLS_TO_READ = ['price_1_ask', 'price_1_bid']
    HORIZONS_SEC = [60, 300, 900,
                    1800]  # 5min, 15min, 30min - less than a minute not useful for realized volatility as we calculate on 5 min returns

    print("### Calculating mid price realized volatility... ###")

    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # calculate mid price
    data = calculate_mid_price(data)
    print("Mid price calculated.")

    # calculate mid price returns for 5 minute horizon
    data = calculate_returns(data, value_col='mid_price', horizons_min=[5])
    print("Past mid price differences calculated.")

    # calculate realized volatility of mid price returns
    data = create_time_based_realized_volatility(data, target_col='mid_price_return_prev_5min',
                                                 windows_seconds=HORIZONS_SEC)

    # Save only the new feature columns along with ID columns
    save_features(data[ID_COLUMNS + [f'mid_price_return_prev_5min_RV_{horizon}s' for horizon in HORIZONS_SEC]],
                  f"{FEATURE_NAME}.parquet")


# In[29]:


# analyze realized volatility feature
analyze_feature_compact(data, 'mid_price_return_5min_RV_900s')
analyze_feature_compact(data, 'mid_price_return_5min_RV_900s_ds')


# ## bid-ask spread

# In[30]:


# calculate bid-ask spread
# constraints:
# - bid-ask spread is defined as best_ask_price - best_bid_price
# - if either price is missing, spread is 0

FEATURE_NAME = "bid_ask_spread"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    COLS_TO_READ = ['price_1_ask', 'price_1_bid']
    ROLLING_HORIZONS_SEC = [30, 60, 180, 300, 900, 1800]  # 30s, 1min, 3min, 5min, 10min, 30min

    print("### Calculating bid-ask spread features... ###")

    # load data
    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # calculate bid-ask spread feature
    data = calculate_bid_ask_spread(data)
    print("Bid-ask spread calculated.")

    # calculate bid-ask spread rolling means
    data = create_time_based_rolling_means(data, 'bid_ask_spread', ROLLING_HORIZONS_SEC)
    print("Bid-ask spread rolling means calculated.")

    # Save only the new feature columns along with ID columns
    save_features(data[ID_COLUMNS + ['bid_ask_spread'] +
                       [f'bid_ask_spread_MA_{window}s' for window in ROLLING_HORIZONS_SEC]], f"{FEATURE_NAME}.parquet")


# In[31]:


# analyze bid-ask spread features
analyze_feature_compact(data, 'bid_ask_spread')
analyze_feature_compact(data, 'bid_ask_spread_MA_300s')
analyze_feature_compact(data, 'bid_ask_spread_MA_300s_ds')


# ## liquidity at best level

# In[32]:


# calculate liquidity at best level per side
# constraints:
# - if cumulative quantity is missing or zero, liquidity is zero

FEATURE_NAME = "liquidity_best_level"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    COLS_TO_READ = ['quantity_1_ask', 'quantity_1_bid']
    print("### Calculating liquidity at best level features... ###")


    def calculate_liquidity_imbalance(df):
        """
        Calculate liquidity imbalance and add it as a new column 'liquidity_imbalance' to the DataFrame.
        Liquidity imbalance is defined as (liquidity_best_bid - liquidity_best_ask) / (liquidity_best_bid + liquidity_best_ask).
        If both sides have zero liquidity, imbalance is set to 0.
        """
        bid_liquidity = df['liquidity_best_bid'].fillna(0)
        ask_liquidity = df['liquidity_best_ask'].fillna(0)
        total_liquidity = bid_liquidity + ask_liquidity
        df['liquidity_imbalance'] = np.where(
            total_liquidity > 0,
            (bid_liquidity - ask_liquidity) / total_liquidity,
            0  # both sides have zero liquidity
        ).astype('float32')
        return df


    # load data
    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # create cols for liquidity at best level per side
    data['liquidity_best_ask'] = data['quantity_1_ask']
    data['liquidity_best_bid'] = data['quantity_1_bid']
    print("Liquidity at best level per side calculated.")

    # calculate liquidity imbalance
    data = calculate_liquidity_imbalance(data)
    print("Liquidity imbalance calculated.")

    # Save only the new feature columns along with ID columns
    save_features(data[ID_COLUMNS + ['liquidity_best_ask', 'liquidity_best_bid', 'liquidity_imbalance']],
                  f"{FEATURE_NAME}.parquet")


# In[33]:


# analyze liquidity at best level features
# analyze_feature_compact(data, 'liquidity_best_ask')
# analyze_feature_compact(data, 'liquidity_best_ask_ds')
analyze_feature_compact(data, 'liquidity_imbalance')
analyze_feature_compact(data, 'liquidity_imbalance_ds')


# ## orderbook depth features

# In[34]:


# calculate orderbook depth features
# constraints:
# - depth per side is defined as sum of (price_level - mid_price) * quantity per side up to level 10

FEATURE_NAME = "orderbook_depth"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    COLS_TO_READ = [
        'quantity_1_ask', 'quantity_1_bid',
        'quantity_2_ask', 'quantity_2_bid',
        'quantity_3_ask', 'quantity_3_bid',
        'quantity_4_ask', 'quantity_4_bid',
        'quantity_5_ask', 'quantity_5_bid',
        'quantity_6_ask', 'quantity_6_bid',
        'quantity_7_ask', 'quantity_7_bid',
        'quantity_8_ask', 'quantity_8_bid',
        'quantity_9_ask', 'quantity_9_bid',
        'quantity_10_ask', 'quantity_10_bid'
    ]

    ROLLING_HORIZONS_SEC = [30, 60, 180, 300, 900, 1800]  # 30s, 1min, 3min, 5min, 10min, 30min

    print("### Calculating orderbook depth features... ###")


    def calculate_orderbook_depth(df, side):
        """
        Calculate orderbook depth per side and add them as new columns 'orderbook_depth_ask' and 'orderbook_depth_bid' to the DataFrame.
        Depth per side is defined as sum of quantity per side up to level 10.
        """
        df[f'orderbook_depth_{side}'] = sum([df[f'quantity_{level}_{side}'] for level in range(1, 11)])
        return df


    # load data
    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # calculate orderbook depth per side
    data = calculate_orderbook_depth(data, 'ask')
    data = calculate_orderbook_depth(data, 'bid')
    print("Orderbook depth per side calculated.")

    # calculate orderbook depth rolling means
    data = create_time_based_rolling_means(data, 'orderbook_depth_ask', ROLLING_HORIZONS_SEC)
    data = create_time_based_rolling_means(data, 'orderbook_depth_bid', ROLLING_HORIZONS_SEC)
    print("Orderbook depth rolling means calculated.")

    # calculate orderbook depth imbalance
    data['orderbook_depth_imbalance'] = np.where(
        (data['orderbook_depth_bid'] + data['orderbook_depth_ask']) > 0,
        (data['orderbook_depth_bid'] - data['orderbook_depth_ask']) / (
                data['orderbook_depth_bid'] + data['orderbook_depth_ask']),
        0  # both sides have zero depth
    ).astype('float32')
    print("Orderbook depth imbalance calculated.")

    # Save only the new feature columns along with ID columns
    save_features(data[ID_COLUMNS + ['orderbook_depth_ask', 'orderbook_depth_bid', 'orderbook_depth_imbalance'] +
                       [f'orderbook_depth_ask_MA_{window}s' for window in ROLLING_HORIZONS_SEC] +
                       [f'orderbook_depth_bid_MA_{window}s' for window in ROLLING_HORIZONS_SEC]],
                  f"{FEATURE_NAME}.parquet")


# In[35]:


# analyze orderbook depth features
analyze_feature_compact(data, 'orderbook_depth_ask')
analyze_feature_compact(data, 'orderbook_depth_ask_ds')
analyze_feature_compact(data, 'orderbook_depth_imbalance')
analyze_feature_compact(data, 'orderbook_depth_imbalance_ds')


# ## orderbook slope

# In[36]:


# calculate orderbook slope as volume weighted regression

FEATURE_NAME = "orderbook_slope"

if not skip_feature(f"{FEATURE_NAME}.parquet"):

    MAX_LEVEL = 10
    COLS_TO_READ = []
    for level in range(1, MAX_LEVEL + 1):
        COLS_TO_READ += [f'price_{level}_ask', f'price_{level}_bid', f'quantity_{level}_ask', f'quantity_{level}_bid']

    ROLLING_HORIZONS_SEC = [30, 60, 180, 300, 900, 1800]  # 30s, 1min, 3min, 5min, 10min, 30min

    print("### Calculating orderbook slope features... ###")


    def calculate_orderbook_slope_weighted_index(df):
        """
        Berechnet den 'Order Book Slope' nach der Formel (20):
        Volumengewichtete Regression von Preis auf Level-Index.
        """
        # 1. Daten laden (N x 10 Matrizen)
        ask_prices = df[[f'price_{i}_ask' for i in range(1, MAX_LEVEL + 1)]].values
        bid_prices = df[[f'price_{i}_bid' for i in range(1, MAX_LEVEL + 1)]].values
        ask_vols = df[[f'quantity_{i}_ask' for i in range(1, MAX_LEVEL + 1)]].values
        bid_vols = df[[f'quantity_{i}_bid' for i in range(1, MAX_LEVEL + 1)]].values

        # Der Vektor der Level-Indizes (1, 2, ..., 10)
        # Shape (1, 10), wird gebroadcastet auf (N, 10)
        levels = np.arange(1, MAX_LEVEL + 1).reshape(1, -1)

        def calculate_slope(prices, vols):
            # Wichtig: NaNs in Vols mit 0 ersetzen für Berechnungen
            vols_clean = np.nan_to_num(vols, nan=0.0)
            total_vol = np.sum(vols_clean, axis=1)  # Nenner der Mittelwerte

            # Sicherstellen, dass wir nicht durch 0 teilen
            valid_rows = total_vol > 0

            # 1. Gewichtete Mittelwerte (Formel 21)
            # i_bar (Gewichteter Index)
            i_bar = np.sum(vols_clean * levels, axis=1) / total_vol
            i_bar = i_bar[:, np.newaxis]  # Reshape für Broadcasting

            # p_bar (Gewichteter Preis)
            # Beachte: prices kann NaNs haben. Wenn vol=0, ist Preis egal.
            # Wir setzen Preis auf 0 wo vol 0 ist, um nansum korrekt zu nutzen
            prices_safe = np.nan_to_num(prices, nan=0.0)
            p_bar = np.sum(vols_clean * prices_safe, axis=1) / total_vol
            p_bar = p_bar[:, np.newaxis]

            # 2. Regression (Formel 20)
            # Zähler: sum V * (i - i_bar) * (p - p_bar)
            numerator = np.sum(vols_clean * (levels - i_bar) * (prices_safe - p_bar), axis=1)

            # Nenner: sum V * (i - i_bar)^2
            denominator = np.sum(vols_clean * (levels - i_bar) ** 2, axis=1)

            # Slope berechnen
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = numerator / denominator

            # Filter: Wenn Nenner 0 (z.B. nur 1 Level mit Volumen), ist Slope undefiniert
            slope[denominator == 0] = np.nan
            slope[~valid_rows] = np.nan

            return slope.astype('float32')

        # Berechnung
        print("  -> Calculating Ask Slope (Weighted Index)...")
        # Ask Slope ist immer POSITIV (Preise steigen mit Level)
        df['orderbook_slope_ask'] = calculate_slope(ask_prices, ask_vols)

        print("  -> Calculating Bid Slope (Weighted Index)...")
        # Bid Slope ist immer NEGATIV (Preise fallen mit Level)
        # Wir nehmen abs(), um eine konsistente "Steilheit" zu haben
        slope_bid = calculate_slope(bid_prices, bid_vols)
        df['orderbook_slope_bid'] = np.abs(slope_bid)

        return df


    # load data
    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # calculate orderbook slope
    data = calculate_orderbook_slope_weighted_index(data)
    print("Orderbook slope elastic calculated.")

    # delete read columns to save memory
    data.drop(columns=COLS_TO_READ, inplace=True)

    # calculate moving averages of orderbook slope
    data = create_time_based_rolling_means(data, 'orderbook_slope_ask', ROLLING_HORIZONS_SEC)
    data = create_time_based_rolling_means(data, 'orderbook_slope_bid', ROLLING_HORIZONS_SEC)
    print("Orderbook slope moving averages calculated.")

    # Save only the new feature columns along with ID columns
    save_features(data[ID_COLUMNS + ['orderbook_slope_ask', 'orderbook_slope_bid'] +
                       [f'orderbook_slope_ask_MA_{window}s' for window in ROLLING_HORIZONS_SEC] +
                       [f'orderbook_slope_bid_MA_{window}s' for window in ROLLING_HORIZONS_SEC]],
                  f"{FEATURE_NAME}.parquet")


# In[37]:


# analyze orderbook slope features
analyze_feature_compact(data, 'orderbook_slope_ask')
analyze_feature_compact(data, 'orderbook_slope_ask_ds')


# ## temporal context features

# In[38]:


# calculate temporal trading environment features, product nature features

FEATURE_NAME = "temporal_context"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    COLS_TO_READ = []

    print("### Calculating temporal context features... ###")

    # load data
    del data
    data = load_files_with_columns(columns=COLS_TO_READ)

    # Sicherstellen, dass die Spalten im Datetime-Format sind
    data['snapshot_times'] = pd.to_datetime(data['snapshot_times'])
    data['delivery_start'] = pd.to_datetime(data['delivery_start'])

    ### calculate temporal trading environment features

    # minutes to delivery, next quarter hour, next hour: linear features -> no encoding
    data['te_min_to_delivery'] = (data['delivery_start'] - data['snapshot_times']).dt.total_seconds() / 60
    data['te_min_to_next_quarter_hour'] = 15 - (data['snapshot_times'].dt.minute % 15)
    data['te_min_to_next_hour'] = 60 - data['snapshot_times'].dt.minute

    # scale linear features into 0-1 by dividing by max value
    data['te_min_to_delivery'] = (data['te_min_to_delivery'] / 300).astype("float32")
    data['te_min_to_next_quarter_hour'] = (data['te_min_to_next_quarter_hour'] / 15).astype("float32")
    data['te_min_to_next_hour'] = (data['te_min_to_next_hour'] / 60).astype("float32")

    # hour of day: 0-23 -> cyclic -> sin/cos encoding
    data['te_hour_of_day'] = data['snapshot_times'].dt.hour
    data = encode_cyclical_feature(data, 'te_hour_of_day', 24)

    # day of week: 0-6 -> cyclic -> sin/cos encoding
    data["te_day_of_week"] = data['snapshot_times'].dt.dayofweek
    data = encode_cyclical_feature(data, 'te_day_of_week', 7)

    # day of year: 1-365 -> cyclic -> sin/cos encoding
    data["te_day_of_year"] = data['snapshot_times'].dt.dayofyear
    data = encode_cyclical_feature(data, 'te_day_of_year', 365)

    # sidc active: binary -> no encoding
    data['te_is_sidc_active'] = (data['te_min_to_delivery'] > (60 / 300)).astype(int)

    print("Temporal context features calculated.")

    ### calculate product nature features

    # hour of day: 0-23 -> cyclic -> sin/cos encoding
    data["pn_hour_of_day"] = data['delivery_start'].dt.hour
    data = encode_cyclical_feature(data, 'pn_hour_of_day', 24)

    # day of week: 0-6 -> cyclic -> sin/cos encoding
    data["pn_day_of_week"] = data['delivery_start'].dt.dayofweek
    data = encode_cyclical_feature(data, 'pn_day_of_week', 7)

    # day of year: 1-365 -> cyclic -> sin/cos encoding
    data["pn_day_of_year"] = data['delivery_start'].dt.dayofyear
    data = encode_cyclical_feature(data, 'pn_day_of_year', 365.25)

    # is peak: binary -> no encoding
    is_peak_weekday = data['delivery_start'].dt.dayofweek < 5  # Monday=0 to Friday=4
    is_peak_hour = (data['delivery_start'].dt.hour >= 8) & (data['delivery_start'].dt.hour < 20)
    data['pn_is_peak_hour'] = (is_peak_weekday & is_peak_hour).astype(int)

    print("Product nature features calculated.")

    # Save only the new feature columns along with ID columns
    new_feature_cols = [
        # Temporal Trading Environment Features
        'te_hour_of_day_sin', 'te_hour_of_day_cos',
        'te_day_of_week_sin', 'te_day_of_week_cos',
        'te_day_of_year_sin', 'te_day_of_year_cos',
        'te_is_sidc_active',
        'te_min_to_delivery', 'te_min_to_next_quarter_hour', 'te_min_to_next_hour',
        # Product Nature Features
        'pn_hour_of_day_sin', 'pn_hour_of_day_cos',
        'pn_day_of_week_sin', 'pn_day_of_week_cos',
        'pn_day_of_year_sin', 'pn_day_of_year_cos',
        'pn_is_peak_hour'
    ]
    save_features(data[ID_COLUMNS + new_feature_cols], f"{FEATURE_NAME}.parquet")


# ## spillover context
# defined as difference of anomaly scores ot metrics between current and previous products at same point in life cycle
# metrics:
# - spread
# - depth_bid
# - depth_ask
# - slope_bid
# - slope_ask

# In[39]:


# init spillover calculation
FEATURE_NAME = "spillover_diffs"
skip_spill = skip_feature(f"{FEATURE_NAME}.parquet")

del data


# ### spread

# In[40]:


# calculate bid-ask spread spillover diff

if not skip_spill:
    SPREAD_FILE = os.path.join(FEATURES_DIR_SEPARATE, "bid_ask_spread.parquet")
    TARGET_COL = "bid_ask_spread"
    COLS_TO_READ = [TARGET_COL]

    print("### Calculating bid-ask spread spillover diff... ###")

    # load data
    data_spread = load_existing_features_file(columns=COLS_TO_READ, file_path=SPREAD_FILE)

    # calculate diffs
    data_spread = create_spillover_diffs_robust_ttd(data_spread, target_cols=[TARGET_COL],
                                                    neighbor_offset_hours=-1)
    print("Bid-ask spread diff to previous delivery hour calculated.")


# ### depth

# In[41]:


# calculate orderbook depth spillover diff
if not skip_spill:
    DEPTH_FILE = os.path.join(FEATURES_DIR_SEPARATE, "orderbook_depth.parquet")
    TARGET_COLS = ['orderbook_depth_bid', 'orderbook_depth_ask']

    print("### Calculating orderbook depth spillover diff... ###")

    # load data
    data_depth = load_existing_features_file(columns=TARGET_COLS, file_path=DEPTH_FILE)

    # calculate diffs
    data_depth = create_spillover_diffs_robust_ttd(data_depth, target_cols=TARGET_COLS, neighbor_offset_hours=-1)
    print("Orderbook depth diffs to previous delivery hour calculated.")


# ### slope

# In[42]:


# calculate orderbook slope spillover diff
if not skip_spill:
    SLOPE_FILE = os.path.join(FEATURES_DIR_SEPARATE, "orderbook_slope.parquet")
    TARGET_COLS = ['orderbook_slope_bid', 'orderbook_slope_ask']

    print("### Calculating orderbook slope spillover diff... ###")

    # load data
    data_slope = load_existing_features_file(columns=TARGET_COLS, file_path=SLOPE_FILE)

    # calculate diffs
    data_slope = create_spillover_diffs_robust_ttd(data_slope, target_cols=TARGET_COLS, neighbor_offset_hours=-1)
    print("Orderbook slope diffs to previous delivery hour calculated.")


# ### merge, scale and save

# In[43]:


# save spillover features along with ID columns
if not skip_spill:
    print("### Merging and saving spillover diff features... ###")
    # merge all spillover features
    data = data_spread.merge(data_depth, on=ID_COLUMNS, how='left')
    data = data.merge(data_slope, on=ID_COLUMNS, how='left')

    # select new feature columns
    new_feature_cols = [
        f'bid_ask_spread_diff_-1h',
        f'orderbook_depth_bid_diff_-1h', f'orderbook_depth_ask_diff_-1h',
        f'orderbook_slope_bid_diff_-1h', f'orderbook_slope_ask_diff_-1h'
    ]

    save_features(data[ID_COLUMNS + new_feature_cols], f"{FEATURE_NAME}.parquet")


# # label

# ## 5min price movement

# In[44]:


# calculate forward price movement (5min)

FEATURE_NAME = "label_5min_return"

if not skip_feature(f"{FEATURE_NAME}.parquet"):
    COLS_TO_READ = ['price_1_ask', 'price_1_bid']
    HORIZONS_MIN = [5]

    print("### Calculating 5min price movement label... ###")

    # load data
    data = data = load_files_with_columns(columns=COLS_TO_READ)

    # calculate mid price
    data = calculate_mid_price(data)
    print("Mid price calculated.")

    # calculate mid price returns for label horizon
    data = calculate_returns(data, value_col='mid_price', horizons_min=HORIZONS_MIN, direction="future")
    print("Mid price returns for label horizon calculated.")

    print("Using regression label per config, generating vola nomalized returns")
    data['label_5min'] = data[f'mid_price_return_next_5min']

# save label along with ID columns
save_features(data[ID_COLUMNS + ['label_5min']], f"{FEATURE_NAME}.parquet")


# # merging

# In[45]:


# NaN counts per column in merged features:
# mid_price_return_1min               167599
# mid_price_return_5min               471823
# mid_price_return_15min             1224720
# mid_price_return_30min             2351859
# weighted_mid_price_return_1min       77165
# weighted_mid_price_return_5min      376541
# weighted_mid_price_return_15min    1123080
# weighted_mid_price_return_30min    2244545
# mp_wmp_return_diff_1min             167599
# mp_wmp_return_diff_5min             471823
# mp_wmp_return_diff_15min           1224720
# mp_wmp_return_diff_30min           2351859
# mid_price_return_5min_RV_300s       482099
# mid_price_return_5min_RV_900s       465752
# mid_price_return_5min_RV_1800s      453518
# bid_ask_spread_lag_10s               13805
# bid_ask_spread_lag_20s               26744
# bid_ask_spread_lag_30s               39465
# bid_ask_spread_lag_60s               77165
# liquidity_best_ask                   54973
# liquidity_best_bid                   31993
# orderbook_depth_ask                 202407
# orderbook_depth_bid                 182451
# orderbook_depth_ask_lag_10s         211100
# orderbook_depth_bid_lag_10s         190286
# orderbook_slope_ask                  58779
# orderbook_slope_bid                  34665
# orderbook_slope_ask_lag_10s          72382
# orderbook_slope_bid_lag_10s          48288
# orderbook_slope_ask_MA_30s           57317
# orderbook_slope_ask_MA_60s           56470
# orderbook_slope_ask_MA_120s          55345
# orderbook_slope_bid_MA_30s           33212
# orderbook_slope_bid_MA_60s           32400
# orderbook_slope_bid_MA_120s          31341
# dtype: int64
# Rows without any NaNs: 19708814 / 22344778 (88.20%)


# In[46]:


import polars as pl
import os
from tqdm import tqdm

os.environ["POLARS_VERBOSE"] = "1"  # Aktiviert detailliertes Logging


def merge_all_feature_files_polars(feature_file_names, id_columns, output_path):
    """
    Merges feature files using Polars streaming engine with progress tracking.
    """
    feature_file_names = list(feature_file_names)  # Sicherstellen, dass es eine Liste ist
    if not feature_file_names:
        print("No files to merge.")
        return

    print(f"Merging {len(feature_file_names)} files using Polars...")

    # 1. Basis-LazyFrame
    first_path = os.path.join(FEATURES_DIR_SEPARATE, feature_file_names[0])
    merged_lf = pl.scan_parquet(first_path)

    # 2. Schleife zum Aufbau des Query-Plans
    # Das geht sehr schnell, da noch keine Daten fließen
    for i in tqdm(range(1, len(feature_file_names)), desc="Building Join Plan"):
        next_path = os.path.join(FEATURES_DIR_SEPARATE, feature_file_names[i])
        next_lf = pl.scan_parquet(next_path)

        merged_lf = merged_lf.join(
            next_lf,
            on=id_columns,
            how="full",
            coalesce=True
        )

    print("Executing streaming join and saving to disk (this may take a while)...")

    # 3. Ausführen
    # Leider gibt es hier keine native Progress Bar, da Rust übernimmt.
    # Aber es ist viel schneller als Pandas!

    # check if output path exists, if not so, create the directory
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        merged_lf.sink_parquet(output_path)
        print(f"✅ Success! Saved to {output_path}")
    except Exception as e:
        print(f"❌ Error during Polars execution: {e}")


# --- Aufruf ---
if SAVING_STRATEGY == SavingStrategy.DEDICATED:
    # read all file names in FEATURES_DIR
    files = [f for f in os.listdir(FEATURES_DIR_SEPARATE) if f.endswith('.parquet')]

    merge_all_feature_files_polars(
        files,
        ID_COLUMNS,
        FEATURES_FILE_MERGED
    )


# # cleaning
# - clean the data by removing rows containing NaN values

# In[47]:


# load merged file
data = pd.read_parquet(FEATURES_FILE_MERGED)
print(f"Original merged data shape: {data.shape}")


# In[48]:


# analyze NaN values in merged file
# print how many line contain nans in the merged dataframe
nan_counts = data.isna().sum()
print("NaN counts per column in merged features:")
print(nan_counts[nan_counts > 0])

# count rows without any nans
rows_without_nans = data.dropna().shape[0]
total_rows = data.shape[0]
print(f"Rows without any NaNs: {rows_without_nans} / {total_rows} ({(rows_without_nans / total_rows) * 100:.2f}%)")


# In[49]:


# actually clean the data by removing rows with nans
cleaned_data = data.dropna()
print(f"Data shape without NA: {cleaned_data.shape}")


# In[50]:


# clip ttd window to specified range (e.g. 0-300 min) to remove outliers and unrealistic values

if MIN_TTD_MINUTES or MAX_TTD_MINUTES:
    print(f"Clipping TTD window to range: {MIN_TTD_MINUTES} - {MAX_TTD_MINUTES} minutes")
    ttd_minutes = (cleaned_data["delivery_start"] - cleaned_data["snapshot_times"]).dt.total_seconds() / 60
    ttd_min_mask = ttd_minutes >= MIN_TTD_MINUTES if MIN_TTD_MINUTES is not None else True
    ttd_max_mask = ttd_minutes <= MAX_TTD_MINUTES if MAX_TTD_MINUTES is not None else True
    print(
        f"Rows to be clipped based on TTD min: {(~ttd_min_mask).sum()} / {len(cleaned_data)} ({((~ttd_min_mask).sum() / len(cleaned_data)) * 100:.2f}%)")
    print(
        f"Rows to be clipped based on TTD max: {(~ttd_max_mask).sum()} / {len(cleaned_data)} ({((~ttd_max_mask).sum() / len(cleaned_data)) * 100:.2f}%)")
    cleaned_data = cleaned_data[ttd_min_mask & ttd_max_mask]
    print(f"Data shape after clipping TTD window: {cleaned_data.shape}")
else:
    print("No TTD clipping applied.")


# In[51]:


# save cleaned data
cleaned_data.to_parquet(FEATURES_FILE_MERGED_CLEANED)
print("Cleaned data saved.")
del data


# # split

# In[52]:


print("### Splitting data into train, validation and test sets by product... ###")

# load data
df = pd.read_parquet(FEATURES_FILE_MERGED_CLEANED)
print(f"Data shape: {df.shape}")


# In[53]:


# Finde alle einzigartigen Produkt-IDs (in Reihenfolge!)
unique_products = df[PRODUCT_ID_COL].unique()

# Splitte die Produkt-LISTE (nicht den DataFrame)
n_products = len(unique_products)
n_train = int(n_products * TRAIN_SIZE)
n_val = int(n_products * VAL_SIZE)

# sorte die Produkte nach ihrer ID, um Reproduzierbarkeit zu gewährleisten
unique_products = sorted(unique_products)

train_products = unique_products[:n_train]
val_products = unique_products[n_train:n_train + n_val]
test_products = unique_products[n_train + n_val:]

# Filtere den DataFrame basierend auf den Produkt-Listen
train_df = df[df[PRODUCT_ID_COL].isin(train_products)]
val_df = df[df[PRODUCT_ID_COL].isin(val_products)]
test_df = df[df[PRODUCT_ID_COL].isin(test_products)]

print(f"Train shape: {train_df.shape} = {len(train_df) / len(df):.2%} of total")
print(f"Validation shape: {val_df.shape} = {len(val_df) / len(df):.2%} of total")
print(f"Test shape: {test_df.shape} = {len(test_df) / len(df):.2%} of total")

# save splits
os.makedirs(FEATURES_DIR_SPLIT, exist_ok=True)

train_df.to_parquet(TRAIN_FILE)
val_df.to_parquet(VAL_FILE)
test_df.to_parquet(TEST_FILE)
print(f"Saved train set to: {TRAIN_FILE}")
print(f"Saved validation set to: {VAL_FILE}")
print(f"Saved test set to: {TEST_FILE}")


# # scale data

# In[54]:


from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in locals() else Path("../../..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.scaler.Asinh1Scaler import StaticAsinh1Scaler


# In[55]:


# load train set

train_df = pd.read_parquet(TRAIN_FILE)


# In[56]:


# fit scaler on train set

feature_cols = [col for col in train_df.columns if not any(
    keyword in col for keyword in SCALER_FEATURE_BLACKLIST_KEYWORDS + ID_COLUMNS
)]

ttd_col_name = "time_to_delivery_min"

train_df[ttd_col_name] = (
        (pd.to_datetime(train_df['delivery_start']) - pd.to_datetime(
            train_df['snapshot_times'])).dt.total_seconds() / 60
).astype('float32')

# init Scaler
scaler = StaticAsinh1Scaler(
    features_to_scale=feature_cols,
    ttd_col=ttd_col_name,
    ttd_bins=range(0, 301, 10),  # 0 to 300 min in 10 min steps
)

# fit scaler
scaler.fit(train_df)

# save scaler
os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)
scaler.save(SCALER_FILE)


# In[64]:


scaler.profile_df_


# In[65]:


# apply scaler to all splits and save scaled versions
for split_file in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
    print(f"scale and saving split: {split_file}")

    #load split df
    split_df = pd.read_parquet(split_file)

    # add ttd_col
    split_df[ttd_col_name] = (
            (pd.to_datetime(split_df['delivery_start']) - pd.to_datetime(
                split_df['snapshot_times'])).dt.total_seconds() / 60
    ).astype('float32')

    # scale
    split_df = scaler.transform(split_df)

    # remove ttd_col
    split_df.drop(columns=[ttd_col_name], inplace=True)

    # save split
    split_df.to_parquet(split_file)
    print(f"Saved sacled split to: {split_file}")

