import os

import numpy as np
import pandas as pd
from util import data_cleaning, eda


def prepare_bike_data():
    # Define column names for the dataset
    columns = ["Date", "Rented Bike Count", "Hour", "Temperature(C)", "Humidity(%)", "Wind speed (m/s)",
               "Visibility (10m)", "Dew point temperature(C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)",
               "Snowfall (cm)", "Seasons", "Holiday", "Functioning Day"]

    # Construct file path and load data with proper encoding
    file_path = os.path.join(os.path.dirname(__file__), 'SeoulBikeData.csv')
    df = pd.read_csv(file_path, names=columns, header=0, encoding='latin1')

    print("\n=== Seoul Bike Sharing Data Processing ===")

    # Inside prepare_bike_data(), after df = pd.read_csv(...) and before print("\n=== Seoul Bike Sharing Data Processing ===")
    # Parse Date and add new features
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    # Cyclical encoding for Hour
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    # Polynomial feature for Temperature
    df['Temperature_sq'] = df['Temperature(C)'] ** 2


    # Display data types
    data_cleaning.display_dtypes("Seoul Bike Sharing", df)

    # Show missing values heatmap
    eda.plot_null_values(df)

    # Remove rows with missing values
    df = data_cleaning.remove_missing_values(df)

    # Handle outliers using IQR method, excluding Rainfall and Snowfall to preserve non-zero values
    columns_to_process = df.select_dtypes(include=['float64', 'int64']).columns.difference(
        ['Rainfall(mm)', 'Snowfall (cm)'])
    df = data_cleaning.remove_outliers_iqr(df, columns_to_process=columns_to_process)

    # Show correlation matrix (excluding categorical/date columns)
    eda.show_data_correlations(df.drop(columns=['Date', 'Seasons', 'Holiday', 'Functioning Day']), "Seoul Bike Demand")

    # Perform group analysis on selected columns
    eda.compute_group_means(df, ['Seasons', 'Holiday', 'Hour', 'Temperature(C)'], 'Rented Bike Count')

    # Summarize descriptive statistics (excluding categorical/date columns)
    eda.summarize_statistics(df.drop(columns=['Date', 'Seasons', 'Holiday', 'Functioning Day']))

    # Visualize outliers for numeric columns
    eda.visualize_outliers(df)

    # Apply one-hot encoding to categorical columns
    df = data_cleaning.one_hot_encode(['Seasons', 'Holiday', 'Functioning Day'], df)

    # Scale numeric features
    #df = data_cleaning.standardize_data(df)

    print("\n=== Processing Complete ===")

    return df.drop(columns='Date')