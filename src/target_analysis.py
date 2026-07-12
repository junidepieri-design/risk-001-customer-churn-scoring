# target_analysis.py
# Analyzes the target variable distribution, balance and quality.
# Receives the cleaned parquet from EDA as input.
import pandas as pd

class TargetAnalysis:
    '''
    Reads the cleaned dataset and analyzes the target variable.
    Covers distribution, balance, nulls and unique values.
    '''

    def __init__(self, file_path, target_column='y'):
        self.file_path = file_path
        self.target_column = target_column
        self.dataframe = pd.read_parquet(file_path)
        self.target = self.dataframe[target_column]

    def unique_values(self):
        # Unique values and data type of the target
        print(f'\nTARGET — {self.target_column}')
        print(f'{"=" * 40}')
        print(f'Type:          {self.target.dtype}')
        print(f'Unique values: {sorted(self.target.unique())}')

    def null_values(self):
        # Null values in the target column
        total_nulls = self.target.isnull().sum()
        print(f'\nNull values:   {total_nulls}')
        if total_nulls > 0:
            print(f'  ⚠️  Target has null values — review before training')

    def distribution(self):
        # Count and percentage per class
        value_counts = self.target.value_counts()
        total_rows = len(self.target)
        print(f'\nDistribution:')
        for class_value, count in value_counts.items():
            percentage = (count / total_rows) * 100
            print(f'  Class {class_value}:  {count}  ({percentage:.1f}%)')

    def balance_check(self):
        # Alerts if minority class is below 30%
        value_counts = self.target.value_counts()
        total_rows = len(self.target)
        minority_percentage = (value_counts.min() / total_rows) * 100
        print(f'\nBalance check:')
        if minority_percentage < 30:
            print(f'  ⚠️  Imbalanced — minority class at {minority_percentage:.1f}%')
            print(f'  Recommendation: use class_weight or resampling strategy')
        else:
            print(f'  ✅ Balanced — minority class at {minority_percentage:.1f}%')

    def run(self):
        '''
        Runs the full target analysis pipeline in sequence.
        Prints a structured report to the console.
        '''
        print(f'\nTARGET ANALYSIS REPORT')
        print(f'{"=" * 40}')
        print(f'Source: {self.file_path}')
        self.unique_values()
        self.null_values()
        self.distribution()
        self.balance_check()
        print(f'\n{"=" * 40}')
        print('Target analysis complete.')
