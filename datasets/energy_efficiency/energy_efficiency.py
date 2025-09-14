import os
import pandas as pd
from util import data_cleaning, eda


def prepare_energy_data():
    """
    Loads the Energy Efficiency dataset, performs exploratory data analysis,
    removes missing values, and visualizes key insights.
    Returns a cleaned DataFrame with no missing values and all relevant columns.
    """
    # Define column names for the dataset
    headers = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution', 'Heating Load', 'Cooling Load']

    # Construct file path and load data from Excel
    data_file = os.path.join(os.path.dirname(__file__), 'ENB2012_data.xlsx')
    df = pd.read_excel(data_file, sheet_name=0, header=0, names=headers)

    print("\n=== Energy Efficiency Data Analysis ===")

    # Display data types
    data_cleaning.display_dtypes("Energy Efficiency", df)

    # Show missing values heatmap
    eda.plot_null_values(df)

    # Remove rows with missing values (if any)
    df = data_cleaning.remove_missing_values(df)

    # Show correlation matrix (excluding categorical-like columns Orientation, Glazing Area Distribution and secondary target Cooling Load)
    eda.show_data_correlations(df, "Energy Efficiency")

    # Perform group analysis for discrete columns on primary target Heating Load
    eda.compute_group_means(df, ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution'], 'Heating Load')

    # Perform group analysis for discrete columns on primary target Cooling Load
    eda.compute_group_means(df, ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution'], 'Cooling Load')

    # Display descriptive statistics (excluding Orientation, Glazing Area Distribution, Cooling Load)
    eda.summarize_statistics(df)

    # Visualize outliers for numeric columns
    eda.visualize_outliers(df)

    print("\n=== Analysis Complete ===")

    return df