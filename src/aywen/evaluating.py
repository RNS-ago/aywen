
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from typing import List, Dict, Any, Optional, Tuple
from aywen.utils import DtypeManager
from aywen.fire_features import add_elliptical_propagation_speed_to_df
import xgboost as xgb
import logging
from collections.abc import Mapping



logger = logging.getLogger("aywen_logger")


def _best_iter_kwargs(model: Any) -> dict:
    kw = {}
    best_it = getattr(model, "best_iteration", None)
    if isinstance(best_it, int) and best_it >= 0:
        kw["iteration_range"] = (0, best_it + 1)  # xgboost >=1.6
    return kw

def _add_prediction_to_df(
    df: pd.DataFrame,
    model: Any,                     # xgboost Booster or sklearn wrapper
    covariates: Optional[List[str]] = None,
    prediction_col = "prediction",
    mgr: Optional[Any] = None,      # your DtypeManager
    include_shap: bool = False,
) -> pd.DataFrame:
    """
    Add point predictions (and optionally SHAP values) to the input DataFrame.
    """
    #--- Call args (debug) ---
    logger.debug(
        "add_prediction_to_df(df.shape=%s, covariates=%s, include_shap=%s)",
        getattr(df, "shape", None), covariates, include_shap
    )

    # 1) Select & type-cast features in a stable order
    X = df[covariates].copy() if covariates is not None else df.copy()
    covariates = list(X.columns)
    if mgr is not None:
        X = mgr.apply(X, strict=True, include=covariates)

    # 2) DMatrix (categoricals supported if X has category dtypes)
    dnew = xgb.DMatrix(X, enable_categorical=True, feature_names=covariates)

    # 3) TreeSHAP contributions on the margin
    contribs = np.asarray(
        model.predict(dnew, pred_contribs=True, **_best_iter_kwargs(model))
    )  # shape: (n_rows, n_features + 1)

    if contribs.ndim != 2 or contribs.shape[1] != len(covariates) + 1:
        raise ValueError(f"Unexpected contribs shape {contribs.shape} for regression.")

    base_values = contribs[:, -1]          # (n_rows,)
    shap_vals  = contribs[:, :-1]          # (n_rows, n_features)
    pred_margin = base_values + shap_vals.sum(axis=1)

    # 4) Assemble output (keep original df + new columns)
    out = df.copy()
    out[prediction_col] = pred_margin

    if include_shap:
        out["base_value"] = base_values
        shap_cols = pd.DataFrame(
            shap_vals, index=X.index, columns=[f"shap_{c}" for c in covariates]
        )
        out = out.join(shap_cols)

    logger.info("Added %s", [prediction_col] + (["base_value"] if include_shap else []) + (list(shap_cols.columns) if include_shap else []))

    return out

def _add_prediction_interval_to_df(
    df: pd.DataFrame,
    prediction_interval_dict: Dict[str, Any],   # {'lo': scalar|Mapping, 'hi': scalar|Mapping}
    covariates: Optional[List[str]] = None,
    lo_col: str = "pi_lo",
    hi_col: str = "pi_hi",
) -> pd.DataFrame:
    """Single-group helper: add pi_lo / pi_hi to df based on scalars or a mapping keyed by covariate tuples."""
    if not (isinstance(prediction_interval_dict, dict)
            and "lo" in prediction_interval_dict and "hi" in prediction_interval_dict):
        raise ValueError("prediction_interval_dict must be a dict with keys 'lo' and 'hi'.")

    lo_val = prediction_interval_dict["lo"]
    hi_val = prediction_interval_dict["hi"]

    if covariates:
        missing = [c for c in covariates if c not in df.columns]
        if missing:
            raise ValueError(f"Missing covariate columns in df: {missing}")

        # tuple key per row
        keys = pd.Series(
            (t for t in df[covariates].itertuples(index=False, name=None)),
            index=df.index
        )
        lo_series = keys.map(lo_val) if isinstance(lo_val, Mapping) else lo_val
        hi_series = keys.map(hi_val) if isinstance(hi_val, Mapping) else hi_val
    else:
        lo_series, hi_series = lo_val, hi_val  # broadcast scalars

    out = df.copy()
    out[lo_col] = lo_series
    out[hi_col] = hi_series
    return out



def add_prediction_to_df(
    df: pd.DataFrame,
    factor1: str,
    factor2: str,
    models: Dict[Tuple[Any, Any], Any],   # keys like (level1, level2)
    covariates: Optional[List[str]] = None,
    prediction_col: str = "prediction",
    mgr: Optional[Any] = None,
    include_shap: bool = False,
) -> pd.DataFrame:
    """
    Add point predictions (and optionally SHAP values) to the input DataFrame.
    Expects `models` keyed by (factor1_value, factor2_value).
    """
    out = df.copy()

    # Pre-allocate containers
    pred_ser = pd.Series(index=out.index, dtype=float, name=prediction_col)
    shap_cols_store: Dict[str, pd.Series] = {}

    # Group once by (factor1, factor2)
    for (k1, k2), idx in out.groupby([factor1, factor2]).groups.items():
        model = models.get((k1, k2))
        if model is None:
            logger.warning("No model for (%r, %r); skipping %d rows.", k1, k2, len(idx))
            continue

        # Compute predictions/SHAP once for this group
        res = _add_prediction_to_df(
            df=out.loc[idx],
            model=model,
            covariates=covariates,
            prediction_col=prediction_col,
            mgr=mgr,
            include_shap=include_shap,
        )

        # Assign predictions
        pred_ser.loc[idx] = res[prediction_col].values

        # Assign SHAP columns (created lazily)
        if include_shap:
            for col in (c for c in res.columns if c.startswith("shap_")):
                if col not in shap_cols_store:
                    shap_cols_store[col] = pd.Series(index=out.index, dtype=res[col].dtype, name=col)
                shap_cols_store[col].loc[idx] = res[col].values

        logger.info(
            "Added %s%s%s for group (%r, %r).",
            prediction_col,
            ", base_value" if include_shap and "base_value" in res.columns else "",
            f", {sum(c.startswith('shap_') for c in res.columns)} SHAP cols" if include_shap else "",
            k1, k2
        )

    # Join back to frame
    out[prediction_col] = pred_ser
    if include_shap and shap_cols_store:
        out = out.join(pd.DataFrame(shap_cols_store))

    return out






def add_prediction_interval_to_df(
    df: pd.DataFrame,
    factor1: str,
    factor2: str,
    prediction_interval_dicts: Dict[Any, Dict[str, Any]],
    covariates: Optional[List[str]] = None,
    lo_col: str = "pi_lo",
    hi_col: str = "pi_hi",
    strict: bool = False,
    restore: bool = False,
) -> pd.DataFrame:
    """
    Add prediction intervals per (factor1, factor2) group.

    Parameters
    ----------
    prediction_interval_dicts
        Either already normalized with tuple keys:
            {(f1, f2): {'lo': scalar|Mapping[tuple, float], 'hi': ...}, ...}
        or raw JSON-ish dict; in that case pass `normalize=deserialize`
        (your inverse of `serialize`) to convert it.

    restore
        If True, attempt to restore the original structure from the normalized form.
        Example: `restore=True`

    strict
        If True, raise KeyError when a group's dict is missing; else leave NaNs.
    """
    out = df.copy()

    # Ensure columns exist (rows with no dict will remain NaN)
    if lo_col not in out:
        out[lo_col] = np.nan
    if hi_col not in out:
        out[hi_col] = np.nan

    # Use user's normalizer (recommended) or assume already-normalized
    group_dicts = restore(prediction_interval_dicts) if restore else prediction_interval_dicts

    # Basic sanity check to avoid silent mismatches
    for k, v in list(group_dicts.items())[:1]:
        if not (isinstance(k, tuple) and len(k) == 2):
            raise TypeError(
                "prediction_interval_dicts keys must be (factor1, factor2) tuples. "
                "Pass a `restore=True` flag if your keys are serialized."
            )
        if not (isinstance(v, dict) and "lo" in v and "hi" in v):
            raise ValueError("Each group entry must be a dict with 'lo' and 'hi'.")

    # Assign per group
    for (k1, k2), idx in out.groupby([factor1, factor2]).groups.items():
        dct = group_dicts.get((k1, k2))
        if dct is None:
            msg = f"No interval dict for group ({k1!r}, {k2!r}); {len(idx)} rows."
            if strict:
                raise KeyError(msg)
            logger.warning(msg)
            continue

        res = _add_prediction_interval_to_df(
            df=out.loc[idx],
            prediction_interval_dict=dct,
            covariates=covariates,
            lo_col=lo_col,
            hi_col=hi_col,
        )

        out.loc[idx, lo_col] = res[lo_col].values
        out.loc[idx, hi_col] = res[hi_col].values

    logger.debug("Added %s and %s.", lo_col, hi_col)

    return out


# ---- evaluation pipeline -----

def evaluation_pipeline(
        models: Dict[str, Any], 
        pi: Dict[str, Any], 
        df: pd.DataFrame, 
        factor1: str, 
        factor2: str,
        mgr: DtypeManager, 
        pi_covariates: List[str], 
        ratio: float,
        prediction_col: str = "prediction",
        lo_col: str = "pi_lo",
        hi_col: str = "pi_hi"
        ) -> Dict[str, Any]:
    
    logger.important("Starting evaluation pipeline")
    
    # Model evaluation
    out = add_prediction_to_df(
        df=df,
        factor1=factor1,
        factor2=factor2,
        models=models,
        covariates=list(mgr.dtype_map.keys()),
        mgr=mgr,
        prediction_col=prediction_col
    )

    # Evaluate prediction intervals
    out = add_prediction_interval_to_df(
        df=out,
        factor1=factor1,
        factor2=factor2,
        prediction_interval_dicts=pi,
        covariates=pi_covariates,
        lo_col=lo_col,
        hi_col=hi_col
    )

    # Add elliptical propagation speed. Default columns names
    out = add_elliptical_propagation_speed_to_df(
        df=out,
        circular_col=prediction_col,
        lo_circular_col=lo_col,
        hi_circular_col=hi_col,
        ratio=ratio
    )

    return out


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