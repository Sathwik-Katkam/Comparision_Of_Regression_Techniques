import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PowerTransformer
from util import metrics


def _to_1d(y):
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    return arr


def plot_true_vs_predicted(true_vals, pred_vals, algorithm_label):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=true_vals, y=pred_vals)
    min_val = np.nanmin([np.min(true_vals), np.min(pred_vals)])
    max_val = np.nanmax([np.max(true_vals), np.max(pred_vals)])
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{algorithm_label} - True vs Predicted')
    plt.show()


# LOG1P TRANSFORM
def train_log1p_model(train_X, test_X, train_y, test_y):
    results = {}
    X_full = np.vstack([train_X, test_X])
    y_full = _to_1d(np.concatenate([train_y, test_y]))

    # In-sample (reporting)
    in_sample = LinearRegression().fit(X_full, np.log1p(y_full))
    y_pred_in = np.expm1(in_sample.predict(X_full))
    results["Logarithmic Regression (In-Sample)"] = metrics.evaluate_metrics(y_full, y_pred_in)

    # Train/Test
    lr = LinearRegression().fit(train_X, np.log1p(_to_1d(train_y)))
    y_hat_train = np.expm1(lr.predict(train_X))
    y_hat_test = np.expm1(lr.predict(test_X))
    results["Logarithmic Regression (Train)"] = metrics.evaluate_metrics(_to_1d(train_y), y_hat_train)
    results["Logarithmic Regression (Test)"] = metrics.evaluate_metrics(_to_1d(test_y), y_hat_test)

    # Plot
    plot_true_vs_predicted(_to_1d(test_y), y_hat_test, 'Logarithmic Regression')

    # 5-fold CV (on transformed target)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(LinearRegression(), X_full, np.log1p(y_full), cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(LinearRegression(), X_full, np.log1p(y_full), cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    results["Logarithmic Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }
    return lr, results


# SQRT TRANSFORM
def train_sqrt_model(train_X, test_X, train_y, test_y):
    results = {}
    X_full = np.vstack([train_X, test_X])
    y_full = _to_1d(np.concatenate([train_y, test_y]))

    # In-sample
    in_sample = LinearRegression().fit(X_full, np.sqrt(y_full))
    y_pred_in = in_sample.predict(X_full) ** 2
    results["Square Root Regression (In-Sample)"] = metrics.evaluate_metrics(y_full, y_pred_in)

    # Train/Test
    lr = LinearRegression().fit(train_X, np.sqrt(_to_1d(train_y)))
    y_hat_train = lr.predict(train_X) ** 2
    y_hat_test = lr.predict(test_X) ** 2
    results["Square Root Regression (Train)"] = metrics.evaluate_metrics(_to_1d(train_y), y_hat_train)
    results["Square Root Regression (Test)"] = metrics.evaluate_metrics(_to_1d(test_y), y_hat_test)

    # Plot
    plot_true_vs_predicted(_to_1d(test_y), y_hat_test, 'Square Root Regression')

    # 5-fold CV (on sqrt scale)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(LinearRegression(), X_full, np.sqrt(y_full), cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(LinearRegression(), X_full, np.sqrt(y_full), cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    results["Square Root Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }
    return lr, results


# BOX-COX
def train_boxcox_model(train_X, test_X, train_y, test_y):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import PowerTransformer
    from util import metrics

    results = {}

    def _to_1d(y):
        arr = np.asarray(y)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        return arr

    def make_positive(arr):
        arr = _to_1d(arr).astype(float)
        shift = 0.0
        minv = np.min(arr)
        if minv <= 0:
            shift = -minv + 1e-6
            arr = arr + shift
        return arr, shift

    def _boxcox_safe_inverse(pt, z, eps=1e-9):
        """Clip z on the Box–Cox scale so 1 + λ z > 0, then inverse."""
        z = np.asarray(z, dtype=float)
        lam = float(pt.lambdas_[0])
        if abs(lam) < 1e-12:
            # log case
            y = np.exp(z)
        else:
            bound = (-1.0 + eps) / lam
            if lam > 0:
                z = np.maximum(z, bound)
            else:
                z = np.minimum(z, bound)
            y = (lam * z + 1.0) ** (1.0 / lam)
        return y

    X_full = np.vstack([train_X, test_X])
    y_full_raw = _to_1d(np.concatenate([train_y, test_y]))
    y_full_pos, full_shift = make_positive(y_full_raw)

    # In-sample
    pt_full = PowerTransformer(method='box-cox', standardize=False)
    y_full_bc = pt_full.fit_transform(y_full_pos.reshape(-1, 1)).ravel()
    in_sample = LinearRegression().fit(X_full, y_full_bc)
    y_pred_in_bc = in_sample.predict(X_full)
    y_pred_in = _boxcox_safe_inverse(pt_full, y_pred_in_bc) - full_shift
    results["Box-Cox Regression (In-Sample)"] = metrics.evaluate_metrics(y_full_raw, y_pred_in)

    # Train/Test
    y_train_raw = _to_1d(train_y)
    y_test_raw  = _to_1d(test_y)
    y_train_pos, train_shift = make_positive(y_train_raw)
    pt = PowerTransformer(method='box-cox', standardize=False)
    y_train_bc = pt.fit_transform(y_train_pos.reshape(-1, 1)).ravel()

    lr = LinearRegression().fit(train_X, y_train_bc)
    yhat_train_bc = lr.predict(train_X)
    yhat_test_bc  = lr.predict(test_X)

    yhat_train = _boxcox_safe_inverse(pt, yhat_train_bc) - train_shift
    yhat_test  = _boxcox_safe_inverse(pt, yhat_test_bc) - train_shift
    results["Box-Cox Regression (Train)"] = metrics.evaluate_metrics(y_train_raw, yhat_train)
    results["Box-Cox Regression (Test)"]  = metrics.evaluate_metrics(y_test_raw,  yhat_test)

    # Plot
    import seaborn as sns, matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_test_raw, y=yhat_test)
    min_val = np.nanmin([np.min(y_test_raw), np.min(yhat_test)])
    max_val = np.nanmax([np.max(y_test_raw), np.max(yhat_test)])
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Box-Cox Regression - True vs Predicted')
    plt.show()

    # 5-fold CV on BC scale
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(LinearRegression(), X_full, y_full_bc, cv=kf, scoring='neg_mean_squared_error')
    r2  =  cross_val_score(LinearRegression(), X_full, y_full_bc, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    results["Box-Cox Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²':  f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }

    return lr, results


# YEO–JOHNSON
def train_yeojohnson_model(train_X, test_X, train_y, test_y):
    results = {}

    X_full = np.vstack([train_X, test_X])
    y_full = _to_1d(np.concatenate([train_y, test_y]))

    # In-sample
    pt_full = PowerTransformer(method='yeo-johnson', standardize=False)
    y_full_yj = pt_full.fit_transform(y_full.reshape(-1, 1)).ravel()
    in_sample = LinearRegression().fit(X_full, y_full_yj)
    y_pred_in_yj = in_sample.predict(X_full).reshape(-1, 1)
    y_pred_in = pt_full.inverse_transform(y_pred_in_yj).ravel()
    results["Yeo-Johnson Regression (In-Sample)"] = metrics.evaluate_metrics(y_full, y_pred_in)

    # Train/Test
    y_train = _to_1d(train_y)
    y_test = _to_1d(test_y)
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    y_train_yj = pt.fit_transform(y_train.reshape(-1, 1)).ravel()

    lr = LinearRegression().fit(train_X, y_train_yj)
    yhat_train_yj = lr.predict(train_X).reshape(-1, 1)
    yhat_test_yj = lr.predict(test_X).reshape(-1, 1)

    yhat_train = pt.inverse_transform(yhat_train_yj).ravel()
    yhat_test = pt.inverse_transform(yhat_test_yj).ravel()
    results["Yeo-Johnson Regression (Train)"] = metrics.evaluate_metrics(y_train, yhat_train)
    results["Yeo-Johnson Regression (Test)"] = metrics.evaluate_metrics(y_test, yhat_test)

    # Plot
    plot_true_vs_predicted(y_test, yhat_test, 'Yeo-Johnson Regression')

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(LinearRegression(), X_full, y_full_yj, cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(LinearRegression(), X_full, y_full_yj, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    results["Yeo-Johnson Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }
    return lr, results