import warnings

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
from typing import List


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
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


class DataLoader:
    def __init__(self, cfg, train_path: Path = None):
        self.cfg = cfg
        self.train_path = Path(cfg.PATHS["train_file"]) if train_path is None else train_path

    def load_train_data(self, sample_n: int = None, keep_id_cols: bool = False, target_col: str = None,
                        features_blacklist_keywords: List = None) -> Tuple[
        pd.DataFrame, pd.Series]:
        """
        Lädt Trainingsdaten mit produkt-basiertem Subsampling.
        Bewahrt die lokale 10s-Struktur für das Training.
        """
        logging.info(f"Lade Trainingsdaten von {self.train_path}...")
        df = pd.read_parquet(self.train_path).pipe(reduce_mem_usage)
        initial_len = len(df)

        # --- NEU: Produkt-basiertes Subsampling ---
        if sample_n and sample_n < initial_len:
            # 1. Alle einzigartigen Produkte finden
            unique_products = df['delivery_start'].unique()

            # 2. Berechne, wie viele Produkte wir ca. brauchen
            # (Annahme: Produkte sind etwa gleich groß)
            avg_rows_per_product = initial_len / len(unique_products)
            num_products_to_sample = int(sample_n / avg_rows_per_product)

            # 3. Ziehe zufällige Produkte über den gesamten Zeitraum
            # Wir sortieren sie danach wieder, um die Chronologie zu wahren
            np.random.seed(self.cfg.RANDOM_STATE if hasattr(self.cfg, 'RANDOM_STATE') else 42)
            sampled_products = np.random.choice(
                unique_products,
                size=min(num_products_to_sample, len(unique_products)),
                replace=False
            )

            # 4. Filtere den DataFrame
            df = df[df['delivery_start'].isin(sampled_products)].copy()
            # Sortierung sicherstellen für TimeSeriesSplit
            df = df.sort_values(['delivery_start', 'snapshot_times'])

            logging.info(f"Produkt-Subsampling aktiv: {len(sampled_products)} Produkte gewählt. "
                         f"Rows: {len(df):,} (Ziel war {sample_n:,}).")

        # --- Rest bleibt gleich ---
        target_col = self.cfg.TARGET_VARIABLE if target_col is None else target_col
        y = df[target_col]

        blacklist = self.cfg.FEATURE_BLACKLIST_KEYWORDS if features_blacklist_keywords is None else features_blacklist_keywords
        feature_cols = [c for c in df.columns if not any(bad_word in c for bad_word in blacklist)]

        if keep_id_cols:
            # Sicherstellen, dass sie nicht doppelt vorkommen
            for col in ["snapshot_times", "delivery_start"]:
                if col not in feature_cols:
                    feature_cols.append(col)

        X = df[feature_cols]

        # Erzwinge Summation in float64
        y_sum = y.astype('float64').sum()
        scale_pos_weight = (len(y) - y_sum) / y_sum
        logging.info(f"Class Imbalance Ratio: {scale_pos_weight:.4f}")
        logging.info(f"Daten bereit für HPO: X={X.shape}, y={y.shape}")
        return X, y
