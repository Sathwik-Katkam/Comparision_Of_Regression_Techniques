from sklearn.model_selection import train_test_split
from datasets.auto_mpg import auto_mpg
from datasets.energy_efficiency import energy_efficiency
from datasets.seoul_bike_sharing_demand import seoul_bike_sharing_demand
from datasets.boston import boston
from linear_regression import train_linear_model, evaluate_cross_validation


def extract_auto_mpg_features():
    data = auto_mpg.prepare_mpg_data()
    features = data.drop(columns=["brand", "mpg"])
    target = data["mpg"]
    return features, target


def extract_energy_efficiency_features():
    data = energy_efficiency.prepare_energy_data();
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
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model = train_linear_model(train_x, test_x, train_y, test_y)
    cv_results = evaluate_cross_validation(model, features, target)
    print("Cross-Validation Results:", cv_results)


def analyze_energy_efficiency():
    data = extract_energy_efficiency_features()
    print('Energy Efficiency Results: Heating Load')
    features = data.drop(columns=['Heating Load'])
    target = data[['Heating Load']]
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model = train_linear_model(train_x, test_x, train_y, test_y)
    cv_results = evaluate_cross_validation(model, features, target)
    print("Cross-Validation Results for Heating Load:", cv_results)
    print()
    print('Energy Efficiency Results: Cooling Load')
    features_cooling = data.drop(columns=['Cooling Load'])
    target_cooling = data[['Cooling Load']]
    train_x, test_x, train_y, test_y = train_test_split(features_cooling, target_cooling, test_size=0.2, random_state=42)
    model = train_linear_model(train_x, test_x, train_y, test_y)
    cv_results = evaluate_cross_validation(model, features, target)
    print("Cross-Validation Results for Cooling Load:", cv_results)


def analyze_seoul_bike():
    features, target = extract_seoul_bike_features()
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model = train_linear_model(train_x, test_x, train_y, test_y)
    cv_results = evaluate_cross_validation(model, features, target)
    print("Cross-Validation Results:", cv_results)


def analyze_boston_housing():
    """
    Runs linear regression on the Boston Housing dataset and evaluates performance.
    """
    features, target = extract_boston_housing_features()
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model = train_linear_model(train_x, test_x, train_y, test_y)
    cv_results = evaluate_cross_validation(model, features, target)
    print("Cross-Validation Performance:", cv_results)


def execute_all_datasets():
    """
    Executes linear regression analysis for all datasets and prints results.
    """
    print("\n=== Auto MPG Regression Analysis ===")
    analyze_auto_mpg()
    print("\n=== Energy Efficieny Regression Analysis ===")
    # analyze_energy_efficiency()
    print("\n=== Seoul Bike Regression Analysis ===")
    analyze_seoul_bike()
    print("\n=== Boston Housing Regression Analysis ===")
    analyze_boston_housing()


if __name__ == "__main__":
    execute_all_datasets()