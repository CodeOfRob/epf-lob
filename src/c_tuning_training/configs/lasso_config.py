import os
import sys
from pathlib import Path
from sklearn.linear_model import Lasso, LogisticRegression

# Projekt-Root und Base-Config Einbindung
sys.path.append(str(Path(__file__).parent))
from base_config import *

# --- Metadaten ---
MODEL_NAME = "Lasso_LargeScale"
JOB_ID = os.getenv("SLURM_JOB_ID", "local")
PATHS["output_dir"] = BASEPATH / "src/hpo/results"


# --- Suchraum ---
def get_search_space(trial):
    return {
        'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
        'tol': trial.suggest_float('tol', 1e-4, 1e-3, log=True),
        'fit_intercept': True
    }


# --- Modell-Factory ---
def get_model(params, is_regression=None):
    """
    Erstellt Lasso/LogReg optimiert für GROSSE DATENSÄTZE.
    """
    regression_mode = is_regression if is_regression is not None else IS_REGRESSION

    # Gemeinsame Basis-Parameter
    base_params = {
        'random_state': RANDOM_STATE,
    }

    model_params = params.copy()

    if regression_mode:
        model_class = Lasso
        print(f"Factory: Creating Lasso Regression (Alpha={model_params['alpha']:.5f})")
        final_params = {**base_params, **model_params}

    else:
        model_class = LogisticRegression

        alpha = model_params.pop('alpha')
        c_value = 1.0 / (alpha + 1e-9)

        print(f"Factory: Creating LogisticRegression SAGA (C={c_value:.5f}) - Multi-Core optimized")

        mode_params = {
            'penalty': 'l1',
            'solver': 'saga',
            'C': c_value,
            'max_iter': 2000,
            'n_jobs': 16,
            'warm_start': False
        }

        final_params = {**base_params, **mode_params, **model_params}

    return model_class(**final_params)
