import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.auto_mpg import auto_mpg
from datasets.energy_efficiency import energy_efficiency
from datasets.seoul_bike_sharing_demand import seoul_bike_sharing_demand
from datasets.boston import boston
from lasso_regression import train_lasso_model, evaluate_cross_validation
from util import data_cleaning


def extract_auto_mpg_features():
    data = auto_mpg.prepare_mpg_data()
    features = data.drop(columns=["brand", "mpg"])
    target = data["mpg"]
    return features, target


def extract_energy_efficiency_features():
    data = energy_efficiency.prepare_energy_data()
    return data


def extract_seoul_bike_features():
    data = seoul_bike_sharing_demand.prepare_bike_data()
    features = data.drop(columns="Rented Bike Count")
    target = data["Rented Bike Count"]
    return features, target


def extract_boston_housing_features():
    data = boston.prepare_boston_data()
    features = data.drop(columns="MEDV")
    target = data["MEDV"]
    return features, target


def analyze_auto_mpg():
    features, target = extract_auto_mpg_features()
    print("Performance including outliers")
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features, target)
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Z Score")
    features_no_outliers_z = data_cleaning.remove_outliers_zscore(features)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_z, target.loc[features_no_outliers_z.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_z, target.loc[features_no_outliers_z.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Inter Quartile")
    features_no_outliers_iq = data_cleaning.remove_outliers_iqr(features)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_iq, target.loc[features_no_outliers_iq.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_iq, target.loc[features_no_outliers_iq.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)


def analyze_energy_efficiency():
    data = extract_energy_efficiency_features()
    print('Energy Efficiency Results: Heating Load')
    features = data.drop(columns=['Heating Load', 'Cooling Load'])
    target = data['Heating Load']
    print("Performance including outliers")
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features, target)
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Z Score")
    features_no_outliers_z = data_cleaning.remove_outliers_zscore(features)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_z, target.loc[features_no_outliers_z.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_z, target.loc[features_no_outliers_z.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Inter Quartile")
    features_no_outliers_iq = data_cleaning.remove_outliers_iqr(features)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_iq, target.loc[features_no_outliers_iq.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_iq, target.loc[features_no_outliers_iq.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print('\nEnergy Efficiency Results: Cooling Load')
    features_cooling = data.drop(columns=['Heating Load', 'Cooling Load'])
    target_cooling = data['Cooling Load']
    print("Performance including outliers")
    train_x, test_x, train_y, test_y = train_test_split(features_cooling, target_cooling, test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_cooling, target_cooling)
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Z Score")
    features_no_outliers_z = data_cleaning.remove_outliers_zscore(features_cooling)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_z, target_cooling.loc[features_no_outliers_z.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_z, target_cooling.loc[features_no_outliers_z.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Inter Quartile")
    features_no_outliers_iq = data_cleaning.remove_outliers_iqr(features_cooling)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_iq, target_cooling.loc[features_no_outliers_iq.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_iq, target_cooling.loc[features_no_outliers_iq.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)


def analyze_seoul_bike():
    features, target = extract_seoul_bike_features()
    print("Performance including outliers")
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features, target)
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Z Score")
    columns_to_process = features.select_dtypes(include=['float64', 'int64']).columns.difference(['Rainfall(mm)', 'Snowfall (cm)'])
    features_no_outliers_z = data_cleaning.remove_outliers_zscore(features, columns_to_process=columns_to_process)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_z, target.loc[features_no_outliers_z.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_z, target.loc[features_no_outliers_z.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Inter Quartile")
    features_no_outliers_iq = data_cleaning.remove_outliers_iqr(features, columns_to_process=columns_to_process)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_iq, target.loc[features_no_outliers_iq.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_iq, target.loc[features_no_outliers_iq.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)


def analyze_boston_housing():
    features, target = extract_boston_housing_features()
    print("Performance including outliers")
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features, target)
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Z Score")
    features_no_outliers_z = data_cleaning.remove_outliers_zscore(features)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_z, target.loc[features_no_outliers_z.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_z, target.loc[features_no_outliers_z.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)

    print("\nPerformance excluding outliers - Inter Quartile")
    features_no_outliers_iq = data_cleaning.remove_outliers_iqr(features)
    train_x, test_x, train_y, test_y = train_test_split(features_no_outliers_iq, target.loc[features_no_outliers_iq.index], test_size=0.2, random_state=42)
    model, results = train_lasso_model(train_x, test_x, train_y, test_y)
    results["Lasso Regression (5X Validation)"] = evaluate_cross_validation(model, features_no_outliers_iq, target.loc[features_no_outliers_iq.index])
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)


def execute_all_datasets():
    """
    Executes Lasso regression analysis for all datasets and prints results.
    """
    print("\n=== Auto MPG Lasso Regression Analysis ===")
    analyze_auto_mpg()
    print("\n=== Energy Efficiency Lasso Regression Analysis ===")
    analyze_energy_efficiency()
    print("\n=== Seoul Bike Lasso Regression Analysis ===")
    analyze_seoul_bike()
    print("\n=== Boston Housing Lasso Regression Analysis ===")
    analyze_boston_housing()


if __name__ == "__main__":
    execute_all_datasets()