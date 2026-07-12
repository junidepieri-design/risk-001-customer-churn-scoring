# preprocessing.py
# Prepares the dataset for feature engineering and model training.
# Handles categorical encoding and numerical scaling.

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Preprocessing:
    '''
    Reads the cleaned parquet and prepares features for modeling.
    Encodes categorical columns, scales numerical columns.
    Saves the processed dataset for the feature engineering step.
    '''

    def __init__(self, file_path, target_column='y'):
        self.file_path = file_path
        self.target_column = target_column
        self.dataframe = pd.read_parquet(file_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def separate_target(self):
        # Separates target from features before any transformation
        self.target = self.dataframe[self.target_column]
        self.features = self.dataframe.drop(columns=[self.target_column])
        print('  ✅ target separated')

    def encode_categoricals(self):
        # Encodes string columns to integer using LabelEncoder
        categorical_columns = self.features.select_dtypes(include='object').columns

        for column_name in categorical_columns:
            label_encoder = LabelEncoder()
            self.features[column_name] = label_encoder.fit_transform(
                self.features[column_name]
            )
            self.label_encoders[column_name] = label_encoder
            print(f'  ✅ encoded: {column_name}')

    def scale_numericals(self):
        # Scales only original numerical columns, excludes label encoded columns
        encoded_columns = list(self.label_encoders.keys())

        numerical_columns = self.features.select_dtypes(
            include=['int64', 'float64']
        ).columns.difference(encoded_columns)

        self.features[numerical_columns] = self.scaler.fit_transform(
            self.features[numerical_columns]
        )
        print(f'  ✅ scaled: {len(numerical_columns)} numerical columns')

    def reattach_target(self):
        # Reattaches target column after transformations
        self.features[self.target_column] = self.target.values
        print('  ✅ target reattached')

    def save(self, output_path='../data/churn_preprocessed.parquet'):
        # Saves the processed dataset for the next pipeline step
        self.features.to_parquet(output_path, index=False)
        print(f'\n✅ Preprocessed dataset saved to {output_path}')
        print(f'   Total columns: {len(self.features.columns)}')

    def save_artifacts(self, encoders_path='../models/label_encoders.pkl', scaler_path='../models/scaler.pkl'):
        # Saves fitted encoders and scaler for use in prediction
        joblib.dump(self.label_encoders, encoders_path)
        joblib.dump(self.scaler, scaler_path)
        print(f'  ✅ encoders saved to {encoders_path}')
        print(f'  ✅ scaler saved to {scaler_path}')

    def run(self):
        '''
        Runs the full preprocessing pipeline in sequence.
        Encodes categoricals, scales numericals, saves output.
        '''
        print(f'\nPREPROCESSING')
        print(f'{"=" * 40}')
        print(f'Source: {self.file_path}')
        print(f'\nTransformations:')
        self.separate_target()
        self.encode_categoricals()
        self.scale_numericals()
        self.reattach_target()
        self.save()
        self.save_artifacts()
        print(f'\n{"=" * 40}')
        print('Preprocessing complete.')
