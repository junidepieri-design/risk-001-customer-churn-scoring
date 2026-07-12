# explainability.py
# Generates SHAP explanations for the trained XGBoost model.
# Covers global feature importance and individual prediction explanations.

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Explainability:
    '''
    Reads the selected feature parquet and the trained model.
    Generates SHAP values for global and local explainability.
    Saves explanation charts to the outputs folder.
    '''

    def __init__(self, file_path, model_path, target_column='y', test_size=0.2):
        self.file_path = file_path
        self.model_path = model_path
        self.target_column = target_column
        self.test_size = test_size
        self.dataframe = pd.read_parquet(file_path)
        self.model = joblib.load(model_path)
        self.features_test = None
        self.shap_values = None
        self.explainer = None

    def prepare_data(self):
        # Splits dataset and keeps test set for explanation
        target = self.dataframe[self.target_column]
        features = self.dataframe.drop(columns=[self.target_column])

        features_train, features_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=42,
            stratify=target
        )

        self.features_test = features_test
        print('  ✅ data prepared')

    def compute_shap_values(self):
        # Computes SHAP values using TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.features_test)
        print(f'  ✅ SHAP values computed for {len(self.features_test)} samples')

    def plot_summary(self, output_path='../outputs/shap_summary.png'):
        # Global feature importance — mean absolute SHAP value per feature
        fig, axis = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values,
            self.features_test,
            show=False,
            plot_size=None
        )
        plt.title('SHAP Summary — Global Feature Importance', fontsize=13)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f'  ✅ summary plot saved to {output_path}')

    def plot_bar(self, output_path='../outputs/shap_bar.png'):
        # Bar chart of mean absolute SHAP values per feature
        fig, axis = plt.subplots(figsize=(10, 7))
        shap.plots.bar(
            self.shap_values,
            show=False,
            max_display=20
        )
        plt.title('SHAP Feature Importance — Mean Absolute Value', fontsize=13)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f'  ✅ bar plot saved to {output_path}')

    def plot_waterfall(self, sample_index=0, output_path='../outputs/shap_waterfall.png'):
        # Local explanation for a single prediction
        fig, axis = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(
            self.shap_values[sample_index],
            show=False
        )
        plt.title(f'SHAP Waterfall — Sample {sample_index}', fontsize=13)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f'  ✅ waterfall plot saved to {output_path}')

    def top_features(self, top_n=10):
        # Prints top N features by mean absolute SHAP value
        mean_shap_values = pd.Series(
            np.abs(self.shap_values.values).mean(axis=0),
            index=self.features_test.columns
        ).sort_values(ascending=False)

        print(f'\nTOP {top_n} FEATURES BY SHAP VALUE')
        print(f'{"=" * 40}')
        for rank, (feature_name, shap_value) in enumerate(
            mean_shap_values.head(top_n).items(), 1
        ):
            print(f'  {rank:02d}. {feature_name:<40} {shap_value:.4f}')

        return mean_shap_values

    def explain_single(self, sample_index=0):
        # Prints a human-readable explanation for a single prediction
        sample_shap = pd.Series(
            self.shap_values.values[sample_index],
            index=self.features_test.columns
        ).sort_values(key=abs, ascending=False)

        sample_score = self.model.predict_proba(
            self.features_test.iloc[[sample_index]]
        )[0][1]

        top_drivers = sample_shap.head(5)

        print(f'\nSINGLE PREDICTION EXPLANATION — Sample {sample_index}')
        print(f'{"=" * 40}')
        print(f'  Churn probability: {sample_score:.2%}')
        print(f'\n  Top drivers:')
        for feature_name, shap_val in top_drivers.items():
            direction = 'increases' if shap_val > 0 else 'decreases'
            print(f'  {feature_name:<40} {direction} churn risk  ({shap_val:+.4f})')

    def run(self):
        '''
        Runs the full explainability pipeline.
        Generates global and local SHAP explanations.
        '''
        print(f'\nEXPLAINABILITY')
        print(f'{"=" * 40}')
        print(f'Source: {self.file_path}')
        print(f'Model:  {self.model_path}')
        print(f'\nComputing SHAP values:')
        self.prepare_data()
        self.compute_shap_values()
        self.top_features()
        self.explain_single(sample_index=0)
        self.plot_summary()
        self.plot_bar()
        self.plot_waterfall()
        print(f'\n{"=" * 40}')
        print('Explainability complete.')
