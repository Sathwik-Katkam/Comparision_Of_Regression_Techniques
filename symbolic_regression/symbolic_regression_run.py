# symbolic_regression/symbolic_regression_run.py

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.auto_mpg import auto_mpg
from datasets.energy_efficiency import energy_efficiency
from datasets.seoul_bike_sharing_demand import seoul_bike_sharing_demand
from datasets.boston import boston

from symbolic_regression import train_symbolic_model


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
    _, r = train_symbolic_model(train_X, test_X, train_y, test_y, label="Symbolic Regression (Auto MPG)")
    results.update(r)
    print_results_table("Auto MPG - Symbolic Regression", results)


def analyze_energy_efficiency():
    data = extract_energy_efficiency_data()

    # Heating Load
    print("Energy Efficiency Results: Heating Load")
    X = data.drop(columns=['Heating Load']).values
    y = data['Heating Load'].values
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_symbolic_model(train_X, test_X, train_y, test_y, label="Symbolic Regression (Energy - Heating)")
    results.update(r)
    print_results_table("Energy Efficiency - Heating Load (Symbolic)", results)
    print()

    # Cooling Load
    print("Energy Efficiency Results: Cooling Load")
    X = data.drop(columns=['Cooling Load']).values
    y = data['Cooling Load'].values
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_symbolic_model(train_X, test_X, train_y, test_y, label="Symbolic Regression (Energy - Cooling)")
    results.update(r)
    print_results_table("Energy Efficiency - Cooling Load (Symbolic)", results)


def analyze_seoul_bike():
    X, y = extract_seoul_bike_features()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_symbolic_model(train_X, test_X, train_y, test_y, label="Symbolic Regression (Seoul Bike)")
    results.update(r)
    print_results_table("Seoul Bike - Symbolic Regression", results)


def analyze_boston_housing():
    X, y = extract_boston_housing_features()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    _, r = train_symbolic_model(train_X, test_X, train_y, test_y, label="Symbolic Regression (Boston Housing)")
    results.update(r)
    print_results_table("Boston Housing - Symbolic Regression", results)


def print_results_table(title, results):
    print(f"\n=== {title} ===")
    df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(df)


def execute_all_datasets():
    print("\n=== Auto MPG – Symbolic Regression ===")
    analyze_auto_mpg()
    print("\n=== Energy Efficiency – Symbolic Regression ===")
    analyze_energy_efficiency()
    print("\n=== Seoul Bike – Symbolic Regression ===")
    analyze_seoul_bike()
    print("\n=== Boston Housing – Symbolic Regression ===")
    analyze_boston_housing()


if __name__ == "__main__":
    execute_all_datasets()