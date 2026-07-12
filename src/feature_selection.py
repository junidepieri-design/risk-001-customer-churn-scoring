# feature_selection.py
# Selects the most relevant features from the candidate feature matrix.
# Uses tree-based importance to rank and filter features automatically.

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

class FeatureSelection:
    '''
    Reads the enriched parquet and selects the most predictive features.
    Uses RandomForest importance to rank features automatically.
    Saves the selected feature set for the training step.
    '''

    def __init__(self, file_path, target_column='y', max_features=20):
        self.file_path = file_path
        self.target_column = target_column
        self.max_features = max_features
        self.dataframe = pd.read_parquet(file_path)
        self.selected_features = None

    def separate_target(self):
        # Separates features and target before selection
        self.target = self.dataframe[self.target_column]
        self.features = self.dataframe.drop(columns=[self.target_column])
        print('  ✅ target separated')

    def rank_features(self):
        # Trains a RandomForest to rank features by importance
        random_forest = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        random_forest.fit(self.features, self.target)

        feature_importances = pd.Series(
            random_forest.feature_importances_,
            index=self.features.columns
        ).sort_values(ascending=False)

        print(f'  ✅ features ranked by importance')
        return feature_importances

    def select_top_features(self, feature_importances):
        # Selects top N features based on importance ranking
        top_feature_names = feature_importances.head(
            self.max_features
        ).index.tolist()

        self.selected_features = self.features[top_feature_names]

        print(f'  ✅ top {self.max_features} features selected')
        print(f'\nSelected features:')
        for rank, feature_name in enumerate(top_feature_names, 1):
            importance_score = feature_importances[feature_name]
            print(f'  {rank:02d}. {feature_name:<40} {importance_score:.4f}')

    def reattach_target(self):
        # Reattaches target column after selection
        self.selected_features[self.target_column] = self.target.values
        print(f'\n  ✅ target reattached')

    def save(self, output_path='../data/churn_selected.parquet'):
        # Saves the selected feature set for the training step
        self.selected_features.to_parquet(output_path, index=False)
        print(f'\n✅ Selected features saved to {output_path}')
        print(f'   Total features: {len(self.selected_features.columns)}')

    def save_artifacts(self, output_path='../models/selected_features.pkl'):
        # Saves the list of selected feature names for use in prediction
        feature_names = self.selected_features.drop(
            columns=[self.target_column]
        ).columns.tolist()
        joblib.dump(feature_names, output_path)
        print(f'  ✅ selected feature list saved to {output_path}')

    def run(self):
        '''
        Runs the full feature selection pipeline.
        Ranks features by importance and selects the top N.
        '''
        print(f'\nFEATURE SELECTION')
        print(f'{"=" * 40}')
        print(f'Source:       {self.file_path}')
        print(f'Max features: {self.max_features}')
        print(f'\nSelecting features:')
        self.separate_target()
        feature_importances = self.rank_features()
        self.select_top_features(feature_importances)
        self.reattach_target()
        self.save()
        self.save_artifacts()
        print(f'\n{"=" * 40}')
        print('Feature selection complete.')
