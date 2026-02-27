import argparse
import logging
import sys
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import optuna
from optuna.exceptions import StorageInternalError
from sklearn.metrics import mean_absolute_error, roc_auc_score, mean_squared_error

# Projekt-Root zum Pfad hinzufügen
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hpo.hpo_pipeline.data_loader import DataLoader
from src.hpo.hpo_pipeline.util import (
    load_config_module, setup_logging, log_full_config,
    reset_stuck_trials, NpEncoder
)
from src.hpo.hpo_pipeline.training import objective

JOB_ID = os.getenv("SLURM_JOB_ID", "local_run")


def parse_args():
    """Definiert und parst die Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description="Run HPO pipeline.")
    parser.add_argument("config_path", type=str, help="Path to python config file")
    parser.add_argument("--train_file", type=str, help="Override train file path")
    parser.add_argument("--subsample_n", type=int, help="Override number of subsamples")
    parser.add_argument("--thinning_freq_sec", type=int, help="Validation thinning frequency in seconds")
    parser.add_argument("--target_type", choices=["class", "reg"], help="Classification or regression")
    parser.add_argument("--optuna_db", type=str, help="SQLITE Path for Optuna DB")
    parser.add_argument("--model_out", type=str, help="Base path for final Model .joblib and logs")
    return parser.parse_args()


def apply_overrides(cfg, args):
    """Wendet Kommandozeilen-Overrides auf das Config-Objekt an."""
    if not args.config_path:
        raise ValueError("Der Pfad zur Konfigurationsdatei muss angegeben werden.")
    if not args.train_file:
        raise ValueError("Der Pfad zur Trainingsdatei (--train_file) muss angegeben werden.")
    if not args.subsample_n:
        raise ValueError("Die Anzahl der Subsamples (--subsample_n) muss angegeben werden.")
    if not args.thinning_freq_sec:
        raise ValueError("Die Thinning-Frequenz (--thinning_freq_sec) muss angegeben werden.")
    if not args.optuna_db:
        raise ValueError("Der Pfad zur Optuna-Datenbank (--optuna_db) muss angegeben werden.")
    if not args.model_out:
        raise ValueError("Der Ausgabepfad für das Modell (--model_out) muss angegeben werden.")
    if not args.target_type:
        raise ValueError("Der Zieltyp (--target_type) muss angegeben werden.")

    cfg.PATHS["train_file"] = Path(args.train_file)
    cfg.HPO_SAMPLE_N = args.subsample_n
    cfg.EVAL_THINNING_FREQ_SEC = args.thinning_freq_sec
    cfg.PATHS["optuna_db"] = args.optuna_db
    cfg.PATHS["output_dir"] = Path(args.model_out)
    cfg.IS_REGRESSION = (args.target_type == "reg")
    return cfg


def run_hpo_study(study, X, y, cfg):
    """Führt die Optuna-Studie mit Resilienz gegen Datenbank-Fehler aus."""
    finished_trials = [t for t in study.trials if t.state in
                       (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    trials_left = cfg.N_TRIALS - len(finished_trials)

    if trials_left <= 0:
        logging.info("Studie bereits abgeschlossen.")
        return

    logging.info(f"Bereits fertig: {len(finished_trials)}. Starte {trials_left} weitere Trials...")

    metric = mean_absolute_error if cfg.IS_REGRESSION else roc_auc_score
    max_retries = 10

    for i in range(max_retries):
        try:
            study.optimize(
                lambda t: objective(
                    t, X, y, cfg.get_model, cfg.get_search_space,
                    metric, cfg.N_CV_SPLITS, cfg.EVAL_THINNING_FREQ_SEC, cfg.IS_REGRESSION
                ),
                n_trials=trials_left,
                n_jobs=1  # Slurm-Architektur: 1 Trial pro Job (bei 4 Cores)
            )
            break
        except StorageInternalError as e:
            logging.warning(f"Datenbank-Fehler (Versuch {i + 1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(30)
            else:
                raise e


def main():
    # 1. Initialization
    args = parse_args()
    cfg = load_config_module(args.config_path)
    cfg = apply_overrides(cfg, args)

    # 2. Directory & Logging Setup
    out_dir = Path(cfg.PATHS["output_dir"]) / cfg.MODEL_NAME / JOB_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    log_full_config(cfg, vars(args))

    mode_str = "Regression" if cfg.IS_REGRESSION else "Klassifikation"
    logging.info(f"START: HPO Pipeline ({cfg.MODEL_NAME}) - Mode: {mode_str}")

    # 3. Data Loading (HPO Phase)
    # keep_id_cols=True ist kritisch für GroupTimeSeriesSplit (delivery_start)
    loader = DataLoader(cfg)
    logging.info(f"Lade HPO-Daten (Subsample: {cfg.HPO_SAMPLE_N})...")
    X, y = loader.load_train_data(sample_n=cfg.HPO_SAMPLE_N, keep_id_cols=True)

    # 4. Optuna Study Preparation
    storage_url = cfg.PATHS["optuna_db"]
    if "sqlite" in storage_url:
        db_path = Path(storage_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=cfg.MODEL_NAME,
        storage=storage_url,
        direction="minimize" if cfg.IS_REGRESSION else "maximize",
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner()
    )

    reset_stuck_trials(study, cfg.MODEL_NAME)

    # 5. Execute HPO
    try:
        run_hpo_study(study, X, y, cfg)
    except KeyboardInterrupt:
        logging.warning("Durch Benutzer unterbrochen.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fataler Fehler während der Optimierung: {e}", exc_info=True)
        sys.exit(1)

    # 6. Final Refit
    logging.info("=" * 30)
    logging.info(f"HPO beendet. Best Score: {study.best_value:.4f}")
    logging.info("Starte finalen Refit auf dem GESAMTEN Datensatz...")

    # ALLES laden (sample_n=None), ohne IDs für das eigentliche Modell-Fitting
    X_full, y_full = loader.load_train_data(sample_n=None, keep_id_cols=False)

    print("Best params")
    print(study.best_params)

    final_model = cfg.get_model(study.best_params, cfg.IS_REGRESSION)

    start_refit = datetime.now()
    final_model.fit(X_full, y_full)
    logging.info(f"Refit abgeschlossen. Dauer: {datetime.now() - start_refit}")

    # 7. Persistence
    model_path = out_dir / f"{cfg.MODEL_NAME}.joblib"
    params_path = out_dir / "best_params.json"

    joblib.dump(final_model, model_path)
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4, cls=NpEncoder)

    logging.info(f"Modell gespeichert: {model_path}")
    logging.info("HPO Pipeline erfolgreich beendet.")


if __name__ == "__main__":
    main()
