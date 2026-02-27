import os
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Projekt-Root und Base-Config Einbindung
sys.path.append(str(Path(__file__).parent))
from base_config import *

# --- Metadaten ---
MODEL_NAME = "RandomForest"
JOB_ID = os.getenv("SLURM_JOB_ID", "local")
PATHS["output_dir"] = BASEPATH / "src/hpo/results"


# --- Suchraum ---
def get_search_space(trial):
    """
    Definiert den Hyperparameter-Suchraum für den Random Forest.
    Fokus auf Komplexitätskontrolle, um Overfitting auf Rauschen zu vermeiden.
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 35),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': True,
    }


# --- Modell-Factory ---
def get_model(params, is_regression=None):
    """
    Erstellt eine RandomForest Instanz.
    """
    regression_mode = is_regression if is_regression is not None else IS_REGRESSION

    if regression_mode:
        model_class = RandomForestRegressor
        mode_params = {
            'criterion': 'absolute_error',
        }
    else:
        model_class = RandomForestClassifier
        mode_params = {
            'criterion': 'gini',
        }

    # Gemeinsame Parameter
    base_params = {
        'random_state': RANDOM_STATE,
        'n_jobs': max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)) - 2),
        'verbose': 0,
    }

    print(f"Factory: Creating Random Forest {model_class.__name__} (n_jobs=4)")

    # Kombination: Basis + Modus-spezifisch + Optuna-Hyperparameter
    final_params = {**base_params, **mode_params, **params}

    return model_class(**final_params)
