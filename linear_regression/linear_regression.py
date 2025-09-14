import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from util import metrics



def train_linear_model(train_features, test_features, train_target, test_target):
    # Initialize and train the model
    model = LinearRegression()
    model.fit(train_features, train_target)

    # Generate predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate performance metrics
    train_metrics = metrics.evaluate_metrics(train_target, train_predictions)
    test_metrics = metrics.evaluate_metrics(test_target, test_predictions)

    # FIX: use targets, not features
    rmse = np.sqrt(mean_squared_error(train_target, train_predictions))
    rsquared = r2_score(train_target, train_predictions)
    valid_rmse = np.sqrt(mean_squared_error(test_target, test_predictions))
    valid_rsquared = r2_score(test_target, test_predictions)

    print("Train RMSE:", rmse)
    print("Train R-squared:", rsquared)
    print("Test RMSE:", valid_rmse)
    print("Test R-squared:", valid_rsquared)

    # Print results
    print("Training Performance:", train_metrics)
    print("Testing Performance:", test_metrics)

    return model


def evaluate_cross_validation(lr, x, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(lr, x, y, cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(lr, x, y, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    return {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }