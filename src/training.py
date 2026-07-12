# training.py
# Trains an XGBoost model with Bayesian hyperparameter optimization via Optuna.
# Reads search space from config/xgb_params.py — no code changes required to tune.

import pandas as pd
import numpy as np
import optuna
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config.xgb_params import SEARCH_SPACE, OPTUNA_TRIALS

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Training:
    '''
    Reads the selected feature parquet and trains an XGBoost classifier.
    Uses Optuna to find the best hyperparameters via Bayesian optimization.
    Saves the trained model for the evaluation and explainability steps.
    '''

    def __init__(self, file_path, target_column='y', n_splits=5):
        self.file_path = file_path
        self.target_column = target_column
        self.n_splits = n_splits
        self.dataframe = pd.read_parquet(file_path)
        self.best_params = None
        self.model = None

    def separate_target(self):
        # Separates features and target before training
        self.target = self.dataframe[self.target_column]
        self.features = self.dataframe.drop(columns=[self.target_column])
        print('  ✅ target separated')

    def objective(self, trial):
        # Optuna objective function — maximizes AUC-ROC via cross-validation
        params = {
            'max_depth':        trial.suggest_int(
                                    'max_depth',
                                    *SEARCH_SPACE['max_depth']
                                ),
            'learning_rate':    trial.suggest_float(
                                    'learning_rate',
                                    *SEARCH_SPACE['learning_rate'],
                                    log=True
                                ),
            'n_estimators':     trial.suggest_int(
                                    'n_estimators',
                                    *SEARCH_SPACE['n_estimators']
                                ),
            'subsample':        trial.suggest_float(
                                    'subsample',
                                    *SEARCH_SPACE['subsample']
                                ),
            'colsample_bytree': trial.suggest_float(
                                    'colsample_bytree',
                                    *SEARCH_SPACE['colsample_bytree']
                                ),
            'min_child_weight': trial.suggest_int(
                                    'min_child_weight',
                                    *SEARCH_SPACE['min_child_weight']
                                ),
            'gamma':            trial.suggest_float(
                                    'gamma',
                                    *SEARCH_SPACE['gamma']
                                ),
            'scale_pos_weight': (self.target == 0).sum() / (self.target == 1).sum(),
            'eval_metric':      'auc',
            'random_state':     42
        }

        cross_validator = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42
        )

        xgb_model = XGBClassifier(**params)

        auc_scores = cross_val_score(
            xgb_model,
            self.features,
            self.target,
            cv=cross_validator,
            scoring='roc_auc',
            n_jobs=-1
        )

        return auc_scores.mean()

    def optimize(self):
        # Runs Optuna optimization to find best hyperparameters
        print(f'  Running {OPTUNA_TRIALS} trials...')

        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=OPTUNA_TRIALS)

        self.best_params = study.best_params
        self.best_params['scale_pos_weight'] = (
            (self.target == 0).sum() / (self.target == 1).sum()
        )
        self.best_params['eval_metric'] = 'auc'
        self.best_params['random_state'] = 42

        print(f'  ✅ best AUC-ROC: {study.best_value:.4f}')
        print(f'\n  Best parameters:')
        for param_name, param_value in self.best_params.items():
            print(f'    {param_name:<25} {param_value}')

    def train(self):
        # Trains the final model with the best parameters found
        self.model = XGBClassifier(**self.best_params)
        self.model.fit(self.features, self.target)
        print(f'\n  ✅ model trained on full dataset')

    def save(self, output_path='../models/xgb_churn.pkl'):
        # Saves the trained model for evaluation and explainability
        joblib.dump(self.model, output_path)
        print(f'\n✅ Model saved to {output_path}')

    def run(self):
        '''
        Runs the full training pipeline.
        Optimizes hyperparameters via Optuna and trains the final model.
        '''
        print(f'\nTRAINING')
        print(f'{"=" * 40}')
        print(f'Source:  {self.file_path}')
        print(f'Trials:  {OPTUNA_TRIALS}')
        print(f'Folds:   {self.n_splits}')
        print(f'\nOptimizing:')
        self.separate_target()
        self.optimize()
        self.train()
        self.save()
        print(f'\n{"=" * 40}')
        print('Training complete.')
