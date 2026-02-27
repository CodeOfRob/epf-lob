import sys
from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRegressor

# Projekt-Root und Base-Config Einbindung
sys.path.append(str(Path(__file__).parent))
from base_config import *

# --- Metadaten ---
MODEL_NAME = "CatBoostGPU"
JOB_ID = os.getenv("SLURM_JOB_ID", "local")
PATHS["output_dir"] = BASEPATH / "src/hpo/results"


# --- Suchraum ---
def get_search_space(trial):
    """
    Definiert den Hyperparameter-Suchraum für CatBoost GPU.
    Einige Parameter (z.B. border_count) sind auf GPU optimiert/begrenzt.
    """
    return {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
    }


# --- Modell-Factory ---
def get_model(params, is_regression=None):
    """
    Erstellt CatBoost Instanz für GPU-Nutzung.

    WICHTIG laut Architektur-Basis:
    CatBoost benötigt explizite int/category casts für kategorische Features,
    falls diese als Floats (0.0) vorliegen.
    """
    regression_mode = is_regression if is_regression is not None else IS_REGRESSION

    if regression_mode:
        model_class = CatBoostRegressor
        mode_params = {
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
        }
    else:
        model_class = CatBoostClassifier
        mode_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
        }

    # Gemeinsame Parameter (GPU-spezifisch)
    base_params = {
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'allow_writing_files': False,
        'task_type': 'GPU',
        'thread_count': 4,
        'bootstrap_type': 'Bernoulli',
    }

    print(f"Factory: Creating CatBoost {model_class.__name__} (GPU Mode)")

    final_params = {**base_params, **mode_params, **params}

    return model_class(**final_params)
