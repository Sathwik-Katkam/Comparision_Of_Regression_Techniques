import pandas as pd
from datasets.auto_mpg import auto_mpg
from datasets.energy_efficiency import energy_efficiency
from datasets.seoul_bike_sharing_demand import seoul_bike_sharing_demand
from datasets.boston import boston

from feature_selection import run_feature_selection_table


def _print_table(title: str, df: pd.DataFrame):
    print(f"\n=== {title} ===")
    with pd.option_context('display.max_columns', None, 'display.max_colwidth', None):
        print(df)


def run_auto_mpg():
    data = auto_mpg.prepare_mpg_data()   #
    data = data.drop(columns='brand')    # remove brand (categorical)
    X = data.drop(columns='mpg')
    y = data['mpg']
    tbl = run_feature_selection_table(X, y)
    _print_table("Feature Selection – Auto MPG", tbl)
    return tbl


def run_energy_efficiency():
    data = energy_efficiency.prepare_energy_data()   #

    # Heating Load
    X_h = data.drop(columns='Heating Load')
    y_h = data['Heating Load']
    tbl_h = run_feature_selection_table(X_h, y_h)
    _print_table("Feature Selection – Energy Efficiency (Heating)", tbl_h)

    # Cooling Load
    X_c = data.drop(columns='Cooling Load')
    y_c = data['Cooling Load']
    tbl_c = run_feature_selection_table(X_c, y_c)
    _print_table("Feature Selection – Energy Efficiency (Cooling)", tbl_c)

    return {"heating": tbl_h, "cooling": tbl_c}


def run_seoul_bike():
    data = seoul_bike_sharing_demand.prepare_bike_data()   #
    X = data.drop(columns='Rented Bike Count')
    y = data['Rented Bike Count']
    tbl = run_feature_selection_table(X, y)
    _print_table("Feature Selection – Seoul Bike Sharing", tbl)
    return tbl


def run_boston():
    data = boston.prepare_boston_data()   #
    X = data.drop(columns='MEDV')
    y = data['MEDV']
    tbl = run_feature_selection_table(X, y)
    _print_table("Feature Selection – Boston Housing", tbl)
    return tbl


def run(save_csv: bool = False):
    results = {
        "auto_mpg": run_auto_mpg(),
        "energy_efficiency": run_energy_efficiency(),
        "seoul_bike": run_seoul_bike(),
        "boston": run_boston(),
    }

    if save_csv:
        for name, df in results.items():
            if isinstance(df, dict):  # energy_efficiency (heating + cooling)
                for subname, subdf in df.items():
                    subdf.to_csv(f"feature_selection_{name}_{subname}.csv", index=True)
            else:
                df.to_csv(f"feature_selection_{name}.csv", index=True)

    return results


if __name__ == "__main__":
    run(save_csv=True)