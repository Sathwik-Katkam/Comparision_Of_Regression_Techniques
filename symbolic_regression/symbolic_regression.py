import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import KFold, cross_val_score
from util import metrics

from sklearn.utils.validation import check_X_y, check_array
from gplearn.genetic import SymbolicRegressor

# Add _validate_data to SymbolicRegressor if missing
if not hasattr(SymbolicRegressor, "_validate_data"):
    def _compat_validate_data(self, X, y=None, y_numeric=False):
        if y is None:
            Xc = check_array(X, accept_sparse=False, dtype=float)

            if not hasattr(self, "n_features_in_"):
                self.n_features_in_ = Xc.shape[1]
            return Xc, None
        # During fit, set n_features_in_
        Xc, yc = check_X_y(
            X, y,
            accept_sparse=False,
            dtype=float,
            y_numeric=y_numeric
        )
        self.n_features_in_ = Xc.shape[1]
        return Xc, yc

    SymbolicRegressor._validate_data = _compat_validate_data

def _to_2d(X):
    X = np.asarray(X)
    return X if X.ndim == 2 else X.reshape(-1, 1)


def _to_1d(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    return y


def plot_true_vs_predicted(true_vals, pred_vals, label):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=true_vals, y=pred_vals)
    lo = min(np.min(true_vals), np.min(pred_vals))
    hi = max(np.max(true_vals), np.max(pred_vals))
    plt.plot([lo, hi], [lo, hi], '--', color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{label} - True vs Predicted')
    plt.show()


def build_symbolic_regressor(random_state=42):
    """
    Construct a SymbolicRegressor with sensible defaults.
    """
    return SymbolicRegressor(
        population_size=3000,
        generations=30,
        tournament_size=20,
        stopping_criteria=0.001,
        const_range=(-1.0, 1.0),
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        parsimony_coefficient=0.01,
        verbose=1,
        random_state=random_state
    )


def train_symbolic_model(train_X, test_X, train_y, test_y, label="Symbolic Regression"):
    results = {}
    train_X = _to_2d(train_X)
    test_X  = _to_2d(test_X)
    train_y = _to_1d(train_y)
    test_y  = _to_1d(test_y)

    # In-sample fit (on full X,y for reporting only)
    X_full = np.vstack([train_X, test_X])
    y_full = np.concatenate([train_y, test_y])
    in_sample_model = build_symbolic_regressor()
    in_sample_model.fit(X_full, y_full)
    y_pred_in = in_sample_model.predict(X_full)
    results[f"{label} (In-Sample)"] = metrics.evaluate_metrics(y_full, y_pred_in)

    # Train/Test fit (fit on train, evaluate train & test)
    model = build_symbolic_regressor()
    model.fit(train_X, train_y)
    y_pred_train = model.predict(train_X)
    y_pred_test  = model.predict(test_X)
    results[f"{label} (Train)"] = metrics.evaluate_metrics(train_y, y_pred_train)
    results[f"{label} (Test)"]  = metrics.evaluate_metrics(test_y,  y_pred_test)

    # Plot test true vs predicted
    plot_true_vs_predicted(test_y, y_pred_test, label)

    # 5-fold cross validation (on full data)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(build_symbolic_regressor(), X_full, y_full, cv=kf, scoring='neg_mean_squared_error')
    r2  =  cross_val_score(build_symbolic_regressor(), X_full, y_full, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    results[f"{label} (5X Validation)"] = {
        'MSE':  f"{np.mean(mse):.5f}",
        'R²':   f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }

    return model, results