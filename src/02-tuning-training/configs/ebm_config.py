import os
import sys
from pathlib import Path
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

# Projekt-Root und Base-Config Einbindung
sys.path.append(str(Path(__file__).parent))
from base_config import *

# --- Metadaten ---
MODEL_NAME = "EBM"
JOB_ID = os.getenv("SLURM_JOB_ID", "local")
PATHS["output_dir"] = BASEPATH / "src/hpo/results"


# --- Suchraum ---
def get_search_space(trial):
    """
    Definiert den Hyperparameter-Suchraum für die Explainable Boosting Machine.
    EBMs sind additive Modelle mit Interaktionstermen.
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'interactions': trial.suggest_int('interactions', 0, 10),
        'max_bins': trial.suggest_int('max_bins', 32, 512),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'outer_bags': trial.suggest_int('outer_bags', 4, 16),
        'inner_bags': trial.suggest_int('inner_bags', 0, 4),
    }


# --- Modell-Factory ---
def get_model(params, is_regression=None):
    """
    Erstellt eine EBM Instanz (Classifier oder Regressor).
    """
    regression_mode = is_regression if is_regression is not None else IS_REGRESSION

    if regression_mode:
        model_class = ExplainableBoostingRegressor
    else:
        model_class = ExplainableBoostingClassifier

    base_params = {
        'random_state': RANDOM_STATE,
        'n_jobs': max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)) - 2),
        'validation_size': 0.15,  # Internes early stopping
    }

    print(f"Factory: Creating EBM {model_class.__name__} (n_jobs=4)")

    final_params = {**base_params, **params}

    return model_class(**final_params)
