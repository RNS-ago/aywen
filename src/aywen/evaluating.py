import numpy as np
import pandas as pd
import xgboost as xgb
import logging

logger = logging.getLogger("aywen_logger")


def eval_point_prediction(X_new, model, covariates, mgr):
    # 1) Build ordered, typed rows
    X_new = X_new[covariates].copy()
    X_ordered = mgr.apply(X_new, strict=True, include=covariates).copy()

    # 2) Predict with TreeSHAP (contributions are on the raw margin scale)
    dnew = xgb.DMatrix(X_ordered, enable_categorical=True, feature_names=covariates)

    contribs = model.predict(dnew, pred_contribs=True)            # (n_rows, n_features+1)
    contribs = np.asarray(contribs)

    # Split contributions into SHAP values and the bias term (base value)
    base_values = contribs[:, -1]                                 # (n_rows,)
    shap_vals = contribs[:, :-1]                                  # (n_rows, n_features)

    # 3) Margin prediction per row: base + sum of SHAP values across features
    pred_margin = base_values + shap_vals.sum(axis=1)             # (n_rows,)

    # 4) Return aligned structures (index matches X_ordered)
    shap_df = pd.DataFrame(shap_vals, columns=covariates, index=X_ordered.index)

    # Convenience: if a single row, return scalars and a dict to keep old behavior
    if len(X_ordered) == 1:
        return {
            "prediction": float(pred_margin[0]),
            "base_value": float(base_values[0]),
            "shap_values": shap_df.iloc[0].to_dict(),
        }

    # Multiple rows: return Series/DataFrame per-row
    return {
        "prediction": pd.Series(pred_margin, index=X_ordered.index, name="prediction"),
        "base_value": pd.Series(base_values, index=X_ordered.index, name="base_value"),
        "shap_values": shap_df,
    }

def eval_prediction_interval(X_new, model, covariates):

    assert isinstance(model, dict) and "lo" in model and "hi" in model, \
        "PI model must be a dictionary with keys 'lo' and 'hi'"

    assert all(c in X_new.columns for c in covariates), "All columns in covariates must be present in X_new"
    
    return {
        "lo": model["lo"].get(tuple(X_new[covariates].values[0]), None) if covariates else model["lo"],
        "hi": model["hi"].get(tuple(X_new[covariates].values[0]), None) if covariates else model["hi"]
    }


# --- residual diagnostics functions ---
    

def var_signed(residuals: pd.Series, alpha: float = 0.95, side: str = "upper") -> float:
    r = residuals.to_numpy()
    if side == "upper":   # underprediction tail
        return float(np.quantile(r, alpha))
    else:                 # overprediction tail (negative)
        return float(np.quantile(r, 1 - alpha))

def cvar_abs(residuals: pd.Series, alpha: float = 0.95) -> float:
    abs_res = residuals.abs().to_numpy()
    cutoff = np.quantile(abs_res, alpha)
    tail = abs_res[abs_res >= cutoff]
    return float(np.mean(tail)) if tail.size else 0.0

def cvar_signed(residuals: pd.Series, alpha: float = 0.95, side: str = "upper") -> float:
    r = residuals.to_numpy()
    if side == "upper":   # underprediction (positive residuals)
        cutoff = np.quantile(r, alpha)
        tail = r[r >= cutoff]
    else:                 # overprediction (negative residuals)
        cutoff = np.quantile(r, 1 - alpha)
        tail = -r[r <= cutoff]  # take magnitude
    return float(np.mean(np.abs(tail))) if tail.size else 0.0


def relative_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    # Model MAE
    mae_model = np.mean(np.abs(y_true - y_pred))
    # Baseline MAE (predicting the mean of y)
    mae_baseline = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae_model / mae_baseline


# --- function for groupby apply ---
def residual_diagnostics(group: pd.DataFrame, y_true: str, y_pred: str, cutoff: float) -> pd.Series:
    y = group[y_true]
    yhat = group[y_pred]
    r = y - yhat

    return pd.Series({
        "RMSE": np.sqrt(np.mean(r**2)),
        "MAE": np.mean(np.abs(r)),
        "RelativeMAE": relative_mae(y, yhat),
        "MaxAbsError": np.abs(r).max(),
        "CVaRAbs": cvar_abs(r, cutoff),
        "VaRUnder": var_signed(r, cutoff, "upper"),
        "VaROver": var_signed(r, cutoff, "lower"),
        "CVaRUnder": cvar_signed(r, cutoff, "upper"),
        "CVaROver": cvar_signed(r, cutoff, "lower")
    })


def compute_residual_diagnostics(
    df: pd.DataFrame,
    cols: list[str],
    y_true: str,
    y_pred: str,
    cutoff: float = 0.95
) -> pd.DataFrame:
    """
    Compute residual diagnostics for each group defined by split.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing true and predicted values.
        cols (list[str]): List of column names to group by.
        y_true (str): Column name for true values.
        y_pred (str): Column name for predicted values.
        cutoff (float): Cutoff quantile for VaR and CVaR calculations.

    Returns:
        pd.DataFrame: DataFrame with residual diagnostics for each group.
    """

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, cols=%s, y_true=%s, y_pred=%s, cutoff=%s",
        df.shape, cols, y_true, y_pred, cutoff
    )

    # validate
    for col in [y_true, y_pred] + cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df.")

    # compute diagnostics
    diagnostics = df.groupby(cols, observed=True).apply(
        lambda g: residual_diagnostics(g, y_true, y_pred, cutoff), include_groups=False
    )

    logger.info("Computed residual diagnostics for %d groups", len(diagnostics))

    return diagnostics