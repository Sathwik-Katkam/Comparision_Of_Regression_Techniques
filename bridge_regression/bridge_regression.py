import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score, KFold
from util import metrics




def plot_true_vs_predicted(true_vals, pred_vals, algorithm_label):
    true_vals = np.asarray(true_vals).ravel()
    pred_vals = np.asarray(pred_vals).ravel()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=true_vals, y=pred_vals, color="blue")
    min_val, max_val = np.min(true_vals), np.max(true_vals)
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{algorithm_label} - True vs Predicted')
    plt.tight_layout()
    plt.show()


class BridgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1, q=0.5, lr=1e-4, max_iter=20000, tol=1e-6, fit_intercept=True):
        self.alpha = alpha
        self.q = q
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        # fitted attrs
        self.coef_ = None
        self.intercept_ = 0.0
        self.x_mean_ = None
        self.x_scale_ = None
        self.y_mean_ = 0.0

    def _standardize_X(self, X):
        X = np.asarray(X, dtype=float)
        if self.x_mean_ is None:
            self.x_mean_ = X.mean(axis=0)
        if self.x_scale_ is None:
            self.x_scale_ = X.std(axis=0)
            self.x_scale_[self.x_scale_ == 0.0] = 1.0
        Z = (X - self.x_mean_) / self.x_scale_
        return Z

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        Z = self._standardize_X(X)                 # standardize features
        self.y_mean_ = y.mean()
        yc = y - self.y_mean_                      # center target

        # add intercept column in standardized space
        if self.fit_intercept:
            Zw = np.c_[np.ones(len(Z)), Z]
        else:
            Zw = Z

        n, d = Zw.shape
        w = np.zeros(d)
        prev = np.inf
        eps = 1e-12

        for _ in range(self.max_iter):
            # gradient of squared loss in standardized space
            r = Zw @ w - yc
            grad = (2.0 / n) * (Zw.T @ r)

            # gradient step
            w_tmp = w - self.lr * grad

            # prox-like shrink for 0<q<1 (q defaults to 0.5)
            shrink = self.lr * self.alpha * self.q * np.power(np.abs(w_tmp) + eps, self.q - 1.0)
            w_new = np.sign(w_tmp) * np.maximum(np.abs(w_tmp) - shrink, 0.0)

            # objective for convergence check
            data_fit = np.linalg.norm(Zw @ w_new - yc) ** 2
            reg_term = self.alpha * np.sum(np.power(np.abs(w_new), self.q))
            obj = data_fit + reg_term

            if not np.isfinite(obj):
                break  # numerical safety

            if abs(prev - obj) < self.tol:
                w = w_new
                break

            w = w_new
            prev = obj

        if self.fit_intercept:
            self.intercept_std_ = w[0]    # intercept in standardized space
            self.w_std_ = w[1:]           # coefs in standardized space
        else:
            self.intercept_std_ = 0.0
            self.w_std_ = w

        # Convert standardized-space weights to original feature space:
        self.coef_ = (self.w_std_ / self.x_scale_)
        self.intercept_ = float(self.y_mean_ + self.intercept_std_ - (self.x_mean_ @ self.coef_))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Z = (X - self.x_mean_) / self.x_scale_
        yhat_centered = (self.intercept_std_ + Z @ self.w_std_)
        return yhat_centered + self.y_mean_


def train_bridge_model(train_features, test_features, train_target, test_target, alpha=0.1, q=0.5):
    model = BridgeRegressor(alpha=alpha, q=q, lr=1e-4, max_iter=20000, tol=1e-6, fit_intercept=True)
    model.fit(train_features, train_target)

    train_pred = model.predict(train_features)
    test_pred = model.predict(test_features)

    train_metrics = metrics.evaluate_metrics(train_target, train_pred)
    test_metrics = metrics.evaluate_metrics(test_target, test_pred)

    print("Training Performance:", train_metrics)
    print("Testing Performance:", test_metrics)


    plot_true_vs_predicted(test_target, test_pred, f'Bridge Regression (q={q})')


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