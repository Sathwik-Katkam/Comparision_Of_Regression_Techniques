import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from util import metrics


def plot_true_vs_predicted(true_vals, pred_vals, algorithm_label):
    """
    Plots true vs predicted values.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=true_vals, y=pred_vals, color="blue")
    min_val, max_val = true_vals.min(), true_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{algorithm_label} - True vs Predicted')
    plt.show()


def train_lasso_model(train_features, test_features, train_target, test_target, alpha=0.1):
    """
    Trains a Lasso regression model, evaluates in-sample, train, and test performance,
    and visualizes results. Returns the trained model and results dictionary.
    """
    results = {}

    # In-sample Lasso Regression
    model_in_sample = Lasso(alpha=alpha, max_iter=10000)
    model_in_sample.fit(train_features, train_target)
    y_pred_in_sample = model_in_sample.predict(train_features)
    results["Lasso Regression (In-Sample)"] = metrics.evaluate_metrics(train_target, y_pred_in_sample)

    # Train and Test Lasso Regression
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(train_features, train_target)
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    results["Lasso Regression (Train)"] = metrics.evaluate_metrics(train_target, train_predictions)
    results["Lasso Regression (Test)"] = metrics.evaluate_metrics(test_target, test_predictions)

    # Visualize predictions
    plot_true_vs_predicted(test_target, test_predictions, 'Lasso Regression')

    return model, results


def evaluate_cross_validation(model, features, target):
    """
    Performs 5-fold cross-validation and returns average MSE, R², and RMSE.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(model, features, target, cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(model, features, target, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    return {
        'Mean Squared Error': f"{np.mean(mse):.5f}",
        'R-Squared': f"{np.mean(r2):.5f}",
        'Root Mean Squared Error': f"{np.mean(rmse):.5f}",
    }