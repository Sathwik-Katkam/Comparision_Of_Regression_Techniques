import os
import pandas as pd
from util import data_cleaning, eda


def prepare_boston_data():
    """
    CRIM: Per capita crime rate by town (continuous).
    ZN: Proportion of residential land zoned for lots over 25,000 sq.ft. (continuous, many zeros).
    INDUS: Proportion of non-retail business acres per town (continuous).
    CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise, categorical/binary).
    NOX: Nitric oxides concentration (parts per 10 million, continuous).
    RM: Average number of rooms per dwelling (continuous).
    AGE: Proportion of owner-occupied units built prior to 1940 (continuous).
    DIS: Weighted distances to five Boston employment centers (continuous).
    RAD: Index of accessibility to radial highways (integer, categorical-like with values 1-24).
    TAX: Full-value property-tax rate per $10,000 (continuous).
    PTRATIO: Pupil-teacher ratio by town (continuous).
    B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town (continuous).
    LSTAT: Percentage of lower status population (continuous).
    MEDV: Target - Median value of owner-occupied homes in $1000s (continuous).
    """
    # Specify column names for the dataset
    headers = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

    # Build file path and load data
    data_file = os.path.join(os.path.dirname(__file__), 'boston.csv')
    df = pd.read_csv(data_file, names=headers, header=0)

    print("\n=== Boston Housing Data Analysis ===")

    # Show data types
    data_cleaning.display_dtypes("Boston Housing", df)

    # Visualize missing values
    eda.plot_null_values(df)

    # Remove rows with missing values
    df = data_cleaning.remove_missing_values(df)

    # Display correlation matrix
    eda.show_data_correlations(df, "Boston Housing")

    # Conduct group analysis on categorical columns
    eda.compute_group_means(df, ['CHAS', 'RAD'], 'MEDV')

    # Summarize descriptive statistics
    eda.summarize_statistics(df)

    # Visualize outliers
    eda.visualize_outliers(df)

    print("\n=== Analysis Complete ===")

    return df