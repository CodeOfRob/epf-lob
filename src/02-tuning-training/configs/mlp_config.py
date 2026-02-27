import os
import sys
from pathlib import Path
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Projekt-Root und Base-Config Einbindung
sys.path.append(str(Path(__file__).parent))
from base_config import *

# --- Metadaten (unverändert) ---
MODEL_NAME = "MLP"
JOB_ID = os.getenv("SLURM_JOB_ID", "local")
PATHS["output_dir"] = BASEPATH / "src/hpo/results"


# --- Suchraum (KORRIGIERT) ---
def get_search_space(trial):
    """
    Definiert den Hyperparameter-Suchraum für das MLP.
    Gibt ein Dictionary zurück, das NUR gültige MLP-Parameter enthält.
    """
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        # Optuna merkt sich 'n_units_l{i}', aber wir fügen es nicht zum finalen Dict hinzu
        num_units = trial.suggest_int(f'n_units_l{i}', 16, 128)
        layers.append(num_units)

    # 2. Baue das finale Parameter-Dictionary, das 1:1 an Scikit-learn geht
    #    Hier kommen die temporären Parameter NICHT hinein.
    model_params = {
        'hidden_layer_sizes': tuple(layers),  # Korrekt als Tuple übergeben
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': 'adam',
        'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),  # L2 Regularisierung
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        'max_iter': 500,
    }
    return model_params


# --- Modell-Factory (KORRIGIERT) ---
def get_model(params, is_regression=None):
    """
    Erstellt eine MLP Instance (Classifier oder Regressor).
    Die übergebenen 'params' sind jetzt sauber und passen zur Signatur.
    """
    model_class = MLPRegressor if (is_regression if is_regression is not None else IS_REGRESSION) else MLPClassifier
    
    # Kopie der Parameter, um sie sicher zu bearbeiten
    p = params.copy()
    
    # --- REKONSTRUKTION & BEREINIGUNG ---
    # 1. Prüfen, ob der 'hidden_layer_sizes' Schlüssel fehlt (wie bei deinem 'best_params' Fall)
    if 'hidden_layer_sizes' not in p:
        print("Hinweis: 'hidden_layer_sizes' fehlt. Rekonstruiere Architektur aus Hilfsparametern...")
        
        n_layers = p.get('n_layers', 1) # Default auf 1, falls 'n_layers' fehlt
        layers = []
        for i in range(n_layers):
            # Hole n_units_l0, n_units_l1, ...
            num_units = p.get(f'n_units_l{i}')
            if num_units:
                layers.append(num_units)
        
        # Füge den rekonstruierten Schlüssel hinzu
        p['hidden_layer_sizes'] = tuple(layers)
    
    # 2. Entferne alle Hilfsparameter, die MLPRegressor nicht kennt
    keys_to_remove = [k for k in p if k.startswith('n_layers') or k.startswith('n_units_l')]
    for k in keys_to_remove:
        p.pop(k, None)
    # ------------------------------------

    # Gemeinsame Parameter für Deep Learning Stabilität
    base_params = {
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 10,
        'shuffle': True,
    }


    print(f"Factory: Creating MLP {model_class.__name__}")

    print("Params:", p)

    # Kombination: Basis-Parameter + Optuna-Hyperparameter
    final_params = {**base_params, **p}

    return model_class(**final_params)