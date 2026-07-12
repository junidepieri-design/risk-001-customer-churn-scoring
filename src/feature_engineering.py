# feature_engineering.py
# Generates candidate features using Polynomial Features.
# Creates interactions and powers between numerical columns automatically.

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineering:
    '''
    Reads the preprocessed parquet and generates candidate features.
    Uses PolynomialFeatures to create interactions between numerical columns.
    Works with any tabular dataset — no manual configuration required.
    '''

    def __init__(self, file_path, target_column='y', degree=2):
        self.file_path = file_path
        self.target_column = target_column
        self.degree = degree
        self.dataframe = pd.read_parquet(file_path)
        self.feature_matrix = None

    def separate_target(self):
        # Separates target before feature generation
        self.target = self.dataframe[self.target_column]
        self.features = self.dataframe.drop(columns=[self.target_column])
        print('  ✅ target separated')

    def generate_polynomial_features(self):
        # Generates interactions and powers between all numerical columns
        numerical_columns = self.features.select_dtypes(
        include=['int64', 'float64']
        ).columns

        self.polynomial = PolynomialFeatures(
            degree=self.degree,
            interaction_only=False,
            include_bias=False
        )

        polynomial_array = self.polynomial.fit_transform(
            self.features[numerical_columns]
        )

        polynomial_feature_names = self.polynomial.get_feature_names_out(
            numerical_columns
        )

        polynomial_dataframe = pd.DataFrame(
            polynomial_array,
            columns=polynomial_feature_names,
            index=self.features.index
        )

        # Removes original columns already in the dataset
        new_feature_columns = [
            column for column in polynomial_dataframe.columns
            if column not in self.features.columns
        ]

        self.feature_matrix = pd.concat(
            [self.features, polynomial_dataframe[new_feature_columns]],
            axis=1
        )

        print(f'  ✅ {len(new_feature_columns)} new features generated')

    def reattach_target(self):
        # Reattaches target column after feature generation
        self.feature_matrix[self.target_column] = self.target.values
        print('  ✅ target reattached')

    def save(self, output_path='../data/churn_features.parquet'):
        # Saves the enriched dataset for the feature selection step
        self.feature_matrix.to_parquet(output_path, index=False)
        print(f'\n✅ Feature matrix saved to {output_path}')
        print(f'   Total features: {len(self.feature_matrix.columns)}')

    def save_artifacts(self, output_path='../models/polynomial.pkl'):
        # Saves fitted PolynomialFeatures object for use in prediction
        joblib.dump(self.polynomial, output_path)
        print(f'  ✅ polynomial transformer saved to {output_path}')

    def run(self):
        '''
        Runs the full feature engineering pipeline.
        Generates polynomial features from all numerical columns.
        '''
        print(f'\nFEATURE ENGINEERING')
        print(f'{"=" * 40}')
        print(f'Source: {self.file_path}')
        print(f'Degree: {self.degree}')
        print(f'\nGenerating features:')
        self.separate_target()
        self.generate_polynomial_features()
        self.reattach_target()
        self.save()
        self.save_artifacts()
        print(f'\n{"=" * 40}')
        print('Feature engineering complete.')
