import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.auto_mpg import auto_mpg
from datasets.energy_efficiency import energy_efficiency
from datasets.seoul_bike_sharing_demand import seoul_bike_sharing_demand
from datasets.boston import boston

from transformed_regression.transformed_regression import (
    train_log1p_model,
    train_sqrt_model,
    train_boxcox_model,
    train_yeojohnson_model,
)


# ----- feature/target extractors -----
def extract_auto_mpg_features():
    data = auto_mpg.prepare_mpg_data()
    features = data.drop(columns=["brand", "mpg"])
    target = data["mpg"]
    return features.values, target.values


def extract_energy_efficiency_data():
    return energy_efficiency.prepare_energy_data()


def extract_seoul_bike_features():
    data = seoul_bike_sharing_demand.prepare_bike_data()
    features = data.drop(columns="Rented Bike Count")
    target = data["Rented Bike Count"]
    return features.values, target.values


def extract_boston_housing_features():
    data = boston.prepare_boston_data()
    features = data.drop(columns="MEDV")
    target = data["MEDV"]
    return features.values, target.values


# ----- per-dataset analyzers -----
def analyze_auto_mpg():
    X, y = extract_auto_mpg_features()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_log1p_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_sqrt_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_boxcox_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_yeojohnson_model(train_X, test_X, train_y, test_y); results.update(r)

    print_results_table("Auto MPG - Transformed Regressions", results)


def analyze_energy_efficiency():
    data = extract_energy_efficiency_data()

    # Heating Load
    print("Energy Efficiency Results: Heating Load")
    X = data.drop(columns=['Heating Load']).values
    y = data['Heating Load'].values
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_log1p_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_sqrt_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_boxcox_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_yeojohnson_model(train_X, test_X, train_y, test_y); results.update(r)
    print_results_table("Energy Efficiency - Heating Load (Transformed)", results)
    print()

    # Cooling Load
    print("Energy Efficiency Results: Cooling Load")
    X = data.drop(columns=['Cooling Load']).values
    y = data['Cooling Load'].values
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_log1p_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_sqrt_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_boxcox_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_yeojohnson_model(train_X, test_X, train_y, test_y); results.update(r)
    print_results_table("Energy Efficiency - Cooling Load (Transformed)", results)


def analyze_seoul_bike():
    X, y = extract_seoul_bike_features()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_log1p_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_sqrt_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_boxcox_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_yeojohnson_model(train_X, test_X, train_y, test_y); results.update(r)

    print_results_table("Seoul Bike - Transformed Regressions", results)


def analyze_boston_housing():
    X, y = extract_boston_housing_features()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_log1p_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_sqrt_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_boxcox_model(train_X, test_X, train_y, test_y); results.update(r)
    _, r = train_yeojohnson_model(train_X, test_X, train_y, test_y); results.update(r)

    print_results_table("Boston Housing - Transformed Regressions", results)


def print_results_table(title, results):
    print(f"\n=== {title} ===")
    df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(df)


def execute_all_datasets():
    print("\n=== Auto MPG Transformed Regressions ===")
    analyze_auto_mpg()
    print("\n=== Energy Efficiency Transformed Regressions ===")
    analyze_energy_efficiency()
    print("\n=== Seoul Bike Transformed Regressions ===")
    analyze_seoul_bike()
    print("\n=== Boston Housing Transformed Regressions ===")
    analyze_boston_housing()


if __name__ == "__main__":
    execute_all_datasets()