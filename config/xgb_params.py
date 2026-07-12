# xgb_params.py
# Search space for XGBoost hyperparameter optimization via Optuna.
# Adjust ranges and trials to control optimization depth.

OPTUNA_TRIALS = 50  # increase for more refined search

SEARCH_SPACE = {
    'max_depth':        (3, 10),
    'learning_rate':    (0.01, 0.3),
    'n_estimators':     (100, 1000),
    'subsample':        (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'min_child_weight': (1, 10),
    'gamma':            (0, 5)
}
