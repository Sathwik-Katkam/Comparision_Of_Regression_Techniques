import os
import pandas as pd
from util import data_cleaning, eda


def prepare_mpg_data():
    """
    Loads the Auto MPG dataset, performs exploratory data analysis, transforms car names into brands,
    removes missing values, and visualizes key insights.
    Returns a cleaned DataFrame with no missing values and all relevant columns.
    """
    # Define column names for the dataset
    headers = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
               'acceleration', 'model_year', 'origin', 'car_name']

    # Construct file path and load data
    data_file = os.path.join(os.path.dirname(__file__), 'auto-mpg.data')
    df = load_mpg_data(data_file, headers)

    print("\n=== Auto MPG Data Analysis ===")

    # Display data types
    data_cleaning.display_dtypes("Auto MPG", df)

    # Show missing values heatmap
    eda.plot_null_values(df)

    # Remove rows with missing values
    df = data_cleaning.remove_missing_values(df)

    # Convert car names to brand names
    df = extract_brand_from_car_name(df)
    df = standardize_brands(df)

    # Show correlation matrix (excluding brand column and origin)
    eda.show_data_correlations(df.drop(columns=['brand','origin']), "Auto MPG")

    # Perform group analysis for specified columns
    eda.compute_group_means(df, ['brand', 'cylinders', 'model_year'], 'mpg')

    # Display descriptive statistics (excluding brand and origin column)
    eda.summarize_statistics(df.drop(columns=['brand', 'origin']))

    # Visualize outliers for numeric columns
    eda.visualize_outliers(df)

    return df


def extract_brand_from_car_name(df):
    """
    Extracts the brand name from the car_name column and drops the original car_name column.
    """
    df['brand'] = df['car_name'].apply(lambda name: name.split()[0])
    return df.drop(columns=['car_name'])


def standardize_brands(df):
    """
    Corrects inconsistent brand names in the brand column using a predefined mapping.
    """
    brand_mapping = {
        'chevroelt': 'chevrolet',
        'chevy': 'chevrolet',
        'maxda': 'mazda',
        'mercedes': 'mercedes-benz',
        'vokswagen': 'volkswagen',
        'vw': 'volkswagen',
        'toyouta': 'toyota',
        'hi': 'unknown'  # Placeholder for potential misparsed brand
    }
    df['brand'] = df['brand'].str.lower().map(brand_mapping).fillna(df['brand'])
    return df


def load_mpg_data(file_path, columns):
    """
    Loads the Auto MPG dataset from a file with space-separated values.
    """
    return pd.read_csv(file_path, sep=r'\s+', names=columns, na_values='?')