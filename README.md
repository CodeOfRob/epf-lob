# Model Comparison Electricity Price Forecating on Limit Order book data

End-to-end experimentation pipeline for the master thesis on limit order book modelling. The repo contains data preparation, feature engineering, hyperparameter optimisation, model training, and evaluation (predictive, economic, and explainability) for several models (EBM, CatBoost, RandomForest, MLP, Lasso).

## Repository layout
- `data/`: parquet datasets (raw pivoted order book windows, engineered features, backtests, predictions). Large files are not tracked here. excluded here because of proprietary data and large file sizes.
- `src/a_data_prep/`: scripts and notebooks for converting raw data, technical cleaning, queue aggregation, and pivoting.
- `src/b_feature_eng/`: feature engineering scripts, scalers, and logs; includes purge utilities and Slurm submission files.
- `src/c_tuning_training/`: Optuna-based HPO pipeline, model configs, and training utilities.
- `src/d_eval/`: evaluation code and notebooks (predictive performance, economic performance, explainability).
- `models/`: saved model artefacts (`*.joblib`) and chosen parameter JSON snapshots. excluded here because of proprietary data and large file sizes.
- `requirements.txt`: frozen conda environment used for the thesis experiments.
