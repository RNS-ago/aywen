
import pandas as pd
import xgboost as xgb

def _ensure_ordered_row(X_row: dict, covariates: list, covariates_categorical: list, cat_dtype_map: dict):
    """
    Build a single-row DataFrame with columns exactly as in `covariates`.
    Extra keys in X_row are ignored; missing keys raise a helpful error.
    """
    missing = [c for c in covariates if c not in X_row]
    if missing:
        raise ValueError(f"Missing required covariates: {missing}")
    df = pd.DataFrame([[X_row[c] for c in covariates]], columns=covariates)
    for c in covariates_categorical:
        if c in df.columns:
            df[c] = pd.Categorical(df[c], categories=cat_dtype_map[c])
    return df


def eval_point_prediction(X_new, model, covariates, covariates_categorical, cat_dtype_map):
    
    # 1) Build ordered, typed row
    df_new = _ensure_ordered_row(X_new, covariates, covariates_categorical, cat_dtype_map)

    # 2) Predict with TreeSHAP
    dnew = xgb.DMatrix(df_new, enable_categorical=True, feature_names=covariates)
    contribs = model.predict(dnew, pred_contribs=True).reshape(-1)  # [n_features + 1]

    base_value = float(contribs[-1])
    shap_vals = contribs[:-1]  # length == len(covariates)

    # 3) Margin prediction (apply your link/back-transform if needed)
    pred_margin = base_value + float(shap_vals.sum())

    # 4) Return as aligned structures
    shap_dict = {covariates[i]: shap_vals[i] for i in range(len(covariates))}
    return {
        "prediction": pred_margin,
        "base_value": base_value,
        "shap_values": shap_dict,
    }


def eval_prediction_interval(X_new, model, covariates, covariates_categorical):

    assert isinstance(model, dict) and "lo" in model and "hi" in model, \
        "PI model must be a dictionary with keys 'lo' and 'hi'"
    return model