import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calculate_r2(actual, predicted):
    return r2_score(y_true=actual, y_pred=predicted)


def compute_mse(actual, predicted):
    return mean_squared_error(y_true=actual, y_pred=predicted)


def compute_rmse(actual, predicted):
    return np.sqrt(compute_mse(actual, predicted))


def evaluate_metrics(actual, predicted):
    mse_value = round(compute_mse(actual, predicted), 5)
    r2_value = round(calculate_r2(actual, predicted), 5)
    rmse_value = round(compute_rmse(actual, predicted), 5)

    return {
        'Mean Squared Error': f"{mse_value:.5f}",
        'R-Squared': f"{r2_value:.5f}",
        'Root Mean Squared Error': f"{rmse_value:.5f}",
    }


def plot_actual_vs_predicted(actual_values, predicted_values, algorithm_name):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actual_values, y=predicted_values, color="blue")
    min_val, max_val = actual_values.min(), actual_values.max()
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{algorithm_name} - True vs Predicted Values')
    plt.show()