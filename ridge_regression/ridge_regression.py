import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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


def train_ridge_model(train_features, test_features, train_target, test_target, alpha=1.0):
    """
    Trains a Ridge regression model, evaluates performance, and visualizes results.
    Returns the trained model.
    """
    # Initialize and train the model
    model = Ridge(alpha=alpha)
    model.fit(train_features, train_target)

    # Generate predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate performance metrics
    train_metrics = metrics.evaluate_metrics(train_target, train_predictions)
    test_metrics = metrics.evaluate_metrics(test_target, test_predictions)

    # Print results
    print("Training Performance:", train_metrics)
    print("Testing Performance:", test_metrics)

    # Visualize predictions
    plot_true_vs_predicted(test_target, test_predictions, 'Ridge Regression')

    return model


def evaluate_cross_validation(model, features, target):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(model, features, target, cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(model, features, target, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    return {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }