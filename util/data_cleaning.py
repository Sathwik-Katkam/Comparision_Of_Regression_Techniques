import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def display_dtypes(dataset_name, df):
    print(f"\n{dataset_name} dataset data types")
    print(df.dtypes)


def remove_missing_values(df):
    initial_shape = df.shape
    df = df.dropna()
    final_shape = df.shape
    print(f"\nInitial shape with missing values: {initial_shape}")
    print(f"Final shape without missing values: {final_shape}")
    print(f"Rows with missing values removed: {initial_shape[0] - final_shape[0]}")
    return df


def one_hot_encode(columns, df):
    print(f"\nApplying one-hot encoding to {columns}")
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df


def standardize_data(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def remove_outliers_iqr(df, columns_to_process=None):
    if columns_to_process is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    else:
        numeric_cols = columns_to_process
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]
    return df


def remove_outliers_zscore(df, threshold=3, columns_to_process=None):
    if columns_to_process is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    else:
        numeric_cols = columns_to_process
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        z_scores = (df[col] - mean) / std
        df = df[np.abs(z_scores) <= threshold]
    return df