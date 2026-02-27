import copy
import importlib.util
import json
import logging
import pprint
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna


def load_config_module(path_str):
    path = Path(path_str)
    if not path.exists(): raise FileNotFoundError(f"Config-Datei nicht gefunden: {path}")
    spec = importlib.util.spec_from_file_location("hpo_config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = output_dir / f"{timestamp}.log"

    # Reset handlers if they exist (for notebook compatibility)
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers: root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return log_path


def log_full_config(cfg_module, args_dict=None):
    """Loggt Config-Modul und CLI-Argumente."""
    config_dict = {}
    for key in dir(cfg_module):
        if key.startswith("__"): continue
        value = getattr(cfg_module, key)
        if isinstance(value, (types.ModuleType, types.FunctionType, type)): continue

        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, dict):
            value = copy.deepcopy(value)

            def convert_paths(d):
                for k, v in d.items():
                    if isinstance(v, Path):
                        d[k] = str(v)
                    elif isinstance(v, dict):
                        convert_paths(v)
                return d

            value = convert_paths(value)
        config_dict[key] = value

    logging.info("=" * 20 + " EXPERIMENT KONFIGURATION " + "=" * 20)
    if args_dict:
        logging.info(f"CLI ARGS:\n{pprint.pformat(args_dict, indent=4)}\n")
    logging.info(f"CONFIG:\n{pprint.pformat(config_dict, indent=4)}")
    logging.info("=" * 66)


def reset_stuck_trials(study, model_name):
    logging.info(f"Prüfe auf abgebrochene Trials für {model_name}...")
    stuck_trials = [t for t in study.trials if
                    t.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.FAIL)]
    if not stuck_trials:
        logging.info("Keine abgebrochenen Trials gefunden.")
        return

    logging.warning(f"Gefunden: {len(stuck_trials)} abgebrochene/fehlgeschlagene Trials.")
    existing_params = [t.params for t in study.trials if
                       t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.WAITING)]

    enqueued_count = 0
    for trial in stuck_trials:
        if trial.params in existing_params: continue
        study.enqueue_trial(trial.params)
        existing_params.append(trial.params)
        enqueued_count += 1

    if enqueued_count > 0: logging.info(f"-> {enqueued_count} Trials neu eingeplant.")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super(NpEncoder, self).default(obj)
