import logging

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from .group_timeseries_split import GroupTimeSeriesSplit  # Dein Custom Splitter


def objective(trial, X, y, model_factory, search_space_fn, metric_fn, n_splits, thinning_freq_sec, is_regression=False):
    """
    Modularisierte Optuna Objective Function.

    Parameters:
    -----------
    trial : optuna.Trial
    X, y : Features und Target
    model_factory : Funktion, die mit Parametern ein Modell-Objekt (fit/predict) erstellt
    search_space_fn : Funktion, die 'trial' nimmt und ein Dict mit Hyperparametern zurückgibt
    metric_fn : Die Metrik-Funktion (y_true, y_pred)
    n_splits : Anzahl der CV-Splits
    thinning_freq_sec : Downsampling-Frequenz für das Val-Set
    is_regression : Boolean. Falls True -> predict(), falls False -> predict_proba() für AUC
    """

    # 1. Parameter aus dem Suchraum ziehen und Modell instanziieren
    params = search_space_fn(trial)
    model = model_factory(params, is_regression)

    # 2. Splitter (GroupTimeSeriesSplit für Produktkonsistenz)
    cv = GroupTimeSeriesSplit(n_splits=n_splits)
    groups = X['delivery_start']  # Wir brauchen 'delivery_start' für die Gruppen, aber dürfen es nicht trainieren
    meta_cols = ['delivery_start', 'snapshot_times']  # Spalten, die das Modell NICHT sehen darf

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):

        # --- A. Training Set (Volle Auflösung) ---
        X_train_raw = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # Metadaten entfernen
        X_train = X_train_raw.drop(columns=meta_cols)

        # --- B. Validation Set (Thinned / Downsampled) ---
        X_val_raw = X.iloc[val_idx]
        y_val_raw = y.iloc[val_idx]

        # Thinning anwenden (wie im Eval Loader)
        # Wir erstellen einen temporären DataFrame für die Filterung
        df_val_temp = X_val_raw.copy()
        df_val_temp['_target'] = y_val_raw

        # Anker-Zeit berechnen
        df_val_temp['_anchor'] = df_val_temp['snapshot_times'].dt.floor(f'{thinning_freq_sec}s')

        # Filtern: Erster Snapshot pro Produkt & Anker
        df_val_thinned = df_val_temp.drop_duplicates(subset=['delivery_start', '_anchor'], keep='first')

        # Aufteilen & Metadaten weg
        X_val = df_val_thinned.drop(columns=meta_cols + ['_target', '_anchor'])
        y_val = df_val_thinned['_target']

        # --- Logging (nur beim jedem 10. Trial ausführlich) ---
        train_start = X_train_raw['snapshot_times'].min()
        train_end = X_train_raw['snapshot_times'].max()
        val_start = X_val_raw['snapshot_times'].min()
        val_end = X_val_raw['snapshot_times'].max()
        logging.info(
            f"Fold {fold_idx + 1}: Train {train_start}-{train_end} | Val {val_start}-{val_end} ({len(y_val)} Samples thinned)")
        logging.info(
            f"  Train size: {len(y_train):,}, Positives: {y_train.astype('float64').sum():,} ({y_train.astype('float64').sum() / len(y_train):.2%}%), Products: {X_train_raw['delivery_start'].nunique():,}")
        logging.info(
            f"  Val size (thinned): {len(y_val):,}, Positives: {y_val.astype('float64').sum():,} ({y_val.astype('float64').sum() / len(y_val):.2%}%), Products: {X_val_raw['delivery_start'].nunique():,}")

        # --- C. Fit & Predict ---
        model.fit(X_train, y_train)

        if is_regression:
            # Bei Regression nutzen wir die direkten Werte
            y_pred = model.predict(X_val)
        else:
            # Bei Klassifikation (AUC) brauchen wir die Wahrscheinlichkeiten der positiven Klasse
            # Prüfen ob predict_proba verfügbar ist (Standard bei sklearn Classifiers)
            y_pred = model.predict_proba(X_val)[:, 1]

        # --- D. Scoring ---
        print("Evaluating fold by {}...".format(metric_fn.__name__))
        current_fold_score = metric_fn(y_val, y_pred)
        fold_scores.append(current_fold_score)

        # Zwischenstand an Optuna melden (für Pruning)
        trial.report(np.mean(fold_scores), step=fold_idx)

        if trial.should_prune():
            logging.info(f"Trial {trial.number} pruned at fold {fold_idx}")
            raise optuna.TrialPruned()

    return np.mean(fold_scores)
