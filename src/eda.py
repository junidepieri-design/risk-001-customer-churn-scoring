# eda.py
# Automated exploratory data analysis for tabular datasets.
# Generates a structured summary report of the dataframe.

import pandas as pd

class EDA:
    '''
    Reads a CSV dataset and generates a structured summary.
    Covers shape, nulls, duplicates, column types,
    categorical distributions, and numerical quartiles.
    '''

    def __init__(self, file_path, target_column='y', columns_to_exclude=None):
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)

        # Remove identifier columns passed by the user
        if columns_to_exclude:
            self.dataframe = self.dataframe.drop(
                columns=columns_to_exclude,
                errors='ignore'
            )

        # Rename target column to y
        self.dataframe = self.dataframe.rename(
            columns={target_column: 'y'}
        )

    def shape(self):
        # Total rows and columns in the dataset
        total_rows, total_columns = self.dataframe.shape
        print(f'\nDATASET SHAPE')
        print(f'{"=" * 40}')
        print(f'Rows:    {total_rows}')
        print(f'Columns: {total_columns}')

    def null_values(self):
        # Count of null values per column
        null_counts = self.dataframe.isnull().sum()
        total_nulls = null_counts.sum()
        print(f'\nNULL VALUES')
        print(f'{"=" * 40}')
        if total_nulls == 0:
            print('No null values found.')
        else:
            print(null_counts[null_counts > 0])

    def duplicates(self):
        # Count of fully duplicated rows
        total_duplicates = self.dataframe.duplicated().sum()
        print(f'\nDUPLICATES')
        print(f'{"=" * 40}')
        print(f'{total_duplicates} duplicated rows found.')

    def column_types(self):
        # Data type of each column
        column_type_counts = self.dataframe.dtypes.value_counts()
        print(f'\nCOLUMN TYPES')
        print(f'{"=" * 40}')
        for column_type, count in column_type_counts.items():
            print(f'{str(column_type):<10} {count} columns')

    def categorical_distribution(self):
        # Value counts and percentage for each categorical column
        categorical_columns = self.dataframe.select_dtypes(include='object').columns
        print(f'\nCATEGORICAL DISTRIBUTION')
        print(f'{"=" * 40}')
        for column_name in categorical_columns:
            value_counts = self.dataframe[column_name].value_counts()
            total_values = len(self.dataframe)
            print(f'\n{column_name}:')
            for category_value, count in value_counts.items():
                percentage = (count / total_values) * 100
                print(f'  {str(category_value):<15} {count}  ({percentage:.1f}%)')

    def numerical_summary(self):
        # Quartiles and descriptive stats for each numerical column
        numerical_columns = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        print(f'\nNUMERICAL SUMMARY')
        print(f'{"=" * 40}')
        numerical_stats = self.dataframe[numerical_columns].describe().T
        print(numerical_stats[['count', 'mean', '25%', '50%', '75%', 'max']].to_string())

    def save(self, output_path='../data/churn_clean.parquet'):
        # Saves the cleaned dataframe to parquet
        self.dataframe.to_parquet(output_path, index=False)
        print(f'\n✅ Clean dataset saved to {output_path}')

    def run(self):
        '''
        Runs the full EDA pipeline in sequence.
        Prints a structured summary report to the console.
        '''
        print(f'\nEDA REPORT')
        print(f'{"=" * 40}')
        print(f'Source: {self.file_path}')
        self.shape()
        self.column_types()
        self.null_values()
        self.duplicates()
        self.categorical_distribution()
        self.numerical_summary()
        self.save()
        print(f'\n{"=" * 40}')
        print('EDA complete.')
