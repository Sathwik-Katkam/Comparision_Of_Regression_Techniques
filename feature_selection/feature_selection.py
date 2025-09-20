import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def forward_selection(X: pd.DataFrame, y) -> tuple[list, float, float]:
    """
    Greedy forward feature selection using R^2 as the score.
    Returns (selected_features, r2, adjusted_r2)
    """
    y = np.ravel(y)
    remaining = list(X.columns)
    selected: list[str] = []
    best_r2 = 0.0
    best_model = None

    while remaining:
        candidates = []
        for c in remaining:
            X_new = X[selected + [c]]
            model = LinearRegression().fit(X_new, y)
            score = r2_score(y, model.predict(X_new))
            candidates.append((score, c))

        candidates.sort(reverse=True, key=lambda t: t[0])
        new_r2, best_c = candidates[0]

        if new_r2 > best_r2 + 1e-12:
            selected.append(best_c)
            remaining.remove(best_c)
            best_r2 = new_r2
            best_model = LinearRegression().fit(X[selected], y)
        else:
            break

    if best_model is not None:
        n, p = len(y), len(selected)
        return selected, best_r2, _adjusted_r2(best_r2, n, p)
    return [], np.nan, np.nan


def backward_selection(X: pd.DataFrame, y, significance_level: float = 0.05) -> tuple[list, float, float]:
    """
    Backward elimination using F-test p-values (f_regression).
    Returns (selected_features, r2, adjusted_r2)
    """
    y = np.ravel(y)
    selected = list(X.columns)

    while len(selected) > 0:
        X_new = X[selected]
        # Fit model for R^2 reporting regardless
        LinearRegression().fit(X_new, y)

        # p-values of each current feature
        _, pvals = f_regression(X_new, y)
        if pvals.size == 0 or np.isnan(pvals).all():
            break

        worst_idx = int(np.nanargmax(pvals))
        worst_p = pvals[worst_idx]

        if worst_p > significance_level:
            # remove the feature with largest p-value
            del_feature = selected[worst_idx]
            selected.remove(del_feature)
        else:
            break

    if selected:
        X_sel = X[selected]
        model = LinearRegression().fit(X_sel, y)
        r2 = r2_score(y, model.predict(X_sel))
        n, p = len(y), len(selected)
        return selected, r2, _adjusted_r2(r2, n, p)
    return [], np.nan, np.nan


def stepwise_selection(
        X: pd.DataFrame,
        y,
        significance_level_in: float = 0.05,
        significance_level_out: float = 0.05
) -> tuple[list, float, float]:
    """
    Classic stepwise (forward with p-in, backward with p-out).
    Returns (selected_features, r2, adjusted_r2)
    """
    y = np.ravel(y)
    all_features = X.columns.tolist()
    selected: list[str] = []
    improved = True

    while improved:
        improved = False

        # --- Forward step (pick best with p < SL_in) ---
        remaining = [f for f in all_features if f not in selected]
        if remaining:
            pvals_fwd = pd.Series(index=remaining, dtype=float)
            for f in remaining:
                X_try = X[selected + [f]]
                # p-value for the *added* feature is last in f_regression(X_try, y)
                _, pvals = f_regression(X_try, y)
                pvals_fwd[f] = pvals[-1] if pvals.size else np.nan

            best_feature = pvals_fwd.idxmin()
            best_p = pvals_fwd.min()
            if np.isfinite(best_p) and best_p < significance_level_in:
                selected.append(best_feature)
                improved = True

        # --- Backward step (drop worst with p > SL_out) ---
        while len(selected) > 0:
            X_sel = X[selected]
            _, pvals = f_regression(X_sel, y)
            if pvals.size == 0 or np.isnan(pvals).all():
                break
            worst_idx = int(np.nanargmax(pvals))
            worst_p = pvals[worst_idx]
            if worst_p > significance_level_out:
                del_feature = selected[worst_idx]
                selected.remove(del_feature)
                improved = True
            else:
                break

    if selected:
        X_sel = X[selected]
        model = LinearRegression().fit(X_sel, y)
        r2 = r2_score(y, model.predict(X_sel))
        n, p = len(y), len(selected)
        return selected, r2, _adjusted_r2(r2, n, p)
    return [], np.nan, np.nan


def run_feature_selection_table(X: pd.DataFrame, y, *,
                                sl_in: float = 0.05, sl_out: float = 0.05) -> pd.DataFrame:
    f_feats, f_r2, f_adj = forward_selection(X, y)
    b_feats, b_r2, b_adj = backward_selection(X, y, significance_level=sl_out)
    s_feats, s_r2, s_adj = stepwise_selection(X, y, significance_level_in=sl_in, significance_level_out=sl_out)

    return pd.DataFrame({
        "Selected Features": [f_feats, b_feats, s_feats],
        "R²": [f_r2, b_r2, s_r2],
        "Adjusted R²": [f_adj, b_adj, s_adj],
        "Num Features": [len(f_feats), len(b_feats), len(s_feats)],
    }, index=["Forward Selection", "Backward Selection", "Stepwise Selection"])