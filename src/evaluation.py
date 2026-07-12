# evaluation.py
# Evaluates the trained XGBoost model using banking-standard metrics.
# Covers AUC-ROC, Gini, KS and LIFT curve.

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class Evaluation:
    '''
    Reads the selected feature parquet and the trained model.
    Computes AUC-ROC, Gini, KS statistic and LIFT curve.
    Saves evaluation charts to the outputs folder.
    '''

    def __init__(self, file_path, model_path, target_column='y', test_size=0.2):
        self.file_path = file_path
        self.model_path = model_path
        self.target_column = target_column
        self.test_size = test_size
        self.dataframe = pd.read_parquet(file_path)
        self.model = joblib.load(model_path)
        self.predictions = None
        self.target_test = None

    def prepare_data(self):
        # Splits dataset into train and test for evaluation
        target = self.dataframe[self.target_column]
        features = self.dataframe.drop(columns=[self.target_column])

        features_train, features_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=42,
            stratify=target
        )

        self.target_test = target_test
        self.predictions = self.model.predict_proba(features_test)[:, 1]
        print('  ✅ data prepared')

    def auc_gini(self):
        # Computes AUC-ROC and Gini coefficient
        auc_score = roc_auc_score(self.target_test, self.predictions)
        gini_score = (2 * auc_score) - 1

        print(f'\nMETRICS')
        print(f'{"=" * 40}')
        print(f'  AUC-ROC: {auc_score:.4f}')
        print(f'  Gini:    {gini_score:.4f}')

        return auc_score, gini_score

    def ks_statistic(self):
        # Computes KS statistic — separation between churners and non-churners
        evaluation_dataframe = pd.DataFrame({
            'score': self.predictions,
            'target': self.target_test.values
        }).sort_values('score', ascending=False)

        total_positive = (evaluation_dataframe['target'] == 1).sum()
        total_negative = (evaluation_dataframe['target'] == 0).sum()

        evaluation_dataframe['cumulative_positive'] = (
            evaluation_dataframe['target'] == 1
        ).cumsum() / total_positive

        evaluation_dataframe['cumulative_negative'] = (
            evaluation_dataframe['target'] == 0
        ).cumsum() / total_negative

        evaluation_dataframe['ks'] = abs(
            evaluation_dataframe['cumulative_positive'] -
            evaluation_dataframe['cumulative_negative']
        )

        ks_value = evaluation_dataframe['ks'].max()
        print(f'  KS:      {ks_value:.4f}')

        return evaluation_dataframe, ks_value

    def lift_curve(self, evaluation_dataframe):
        # Computes LIFT per decile
        evaluation_dataframe = evaluation_dataframe.copy()
        evaluation_dataframe['decile'] = pd.qcut(
            evaluation_dataframe['score'],
            q=10,
            labels=False,
            duplicates='drop'
        )

        total_churn_rate = evaluation_dataframe['target'].mean()

        lift_dataframe = evaluation_dataframe.groupby('decile').agg(
            total=('target', 'count'),
            churners=('target', 'sum')
        ).reset_index()

        lift_dataframe['churn_rate'] = (
            lift_dataframe['churners'] / lift_dataframe['total']
        )

        lift_dataframe['lift'] = (
            lift_dataframe['churn_rate'] / total_churn_rate
        )

        lift_dataframe = lift_dataframe.sort_values('decile', ascending=False)

        print(f'\nLIFT BY DECILE')
        print(f'{"=" * 40}')
        print(f'  {"Decile":<10} {"Churners":<12} {"Churn Rate":<14} {"Lift"}')
        for _, lift_row in lift_dataframe.iterrows():
            print(
                f'  {int(lift_row.decile) + 1:<10}'
                f'{int(lift_row.churners):<12}'
                f'{lift_row.churn_rate:.2%}          '
                f'{lift_row.lift:.2f}x'
            )

        return lift_dataframe

    def plot_ks(self, evaluation_dataframe, ks_value, output_path='../outputs/ks_curve.png'):
        # Plots the KS curve
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            evaluation_dataframe['cumulative_positive'].values,
            label='Churners', color='#1B2A4A'
        )
        axis.plot(
            evaluation_dataframe['cumulative_negative'].values,
            label='Non-Churners', color='#8A8880'
        )
        axis.set_title(f'KS Curve — KS = {ks_value:.4f}', fontsize=13)
        axis.set_xlabel('Population %')
        axis.set_ylabel('Cumulative %')
        axis.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f'\n  ✅ KS curve saved to {output_path}')

    def plot_lift(self, lift_dataframe, output_path='../outputs/lift_curve.png'):
        # Plots the LIFT curve by decile
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.bar(
            lift_dataframe['decile'] + 1,
            lift_dataframe['lift'],
            color='#1B2A4A',
            alpha=0.85
        )
        axis.axhline(y=1, color='#8A8880', linestyle='--', label='Baseline')
        axis.set_title('LIFT Curve by Decile', fontsize=13)
        axis.set_xlabel('Decile')
        axis.set_ylabel('Lift')
        axis.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f'  ✅ LIFT curve saved to {output_path}')

    def run(self):
        '''
        Runs the full evaluation pipeline.
        Computes AUC-ROC, Gini, KS and LIFT. Saves charts.
        '''
        print(f'\nEVALUATION')
        print(f'{"=" * 40}')
        print(f'Source: {self.file_path}')
        print(f'Model:  {self.model_path}')
        print(f'\nPreparing:')
        self.prepare_data()
        self.auc_gini()
        evaluation_dataframe, ks_value = self.ks_statistic()
        lift_dataframe = self.lift_curve(evaluation_dataframe)
        self.plot_ks(evaluation_dataframe, ks_value)
        self.plot_lift(lift_dataframe)
        print(f'\n{"=" * 40}')
        print('Evaluation complete.')
