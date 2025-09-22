import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
import json
import logging
from aywen.utils import serialize

logger = logging.getLogger("aywen_logger")

def xgb_tune_fit(X_train, y_train, X_valid, y_valid, base_params):

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    def objective(trial):

        params = {
            'tree_method': trial.suggest_categorical('tree_method', ['hist']),
            'max_depth': trial.suggest_int('max_depth', 4, 12), # The larger the more complex
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50), # The larger the more conservative
            'subsample': trial.suggest_float('subsample', 0.5, 1.0), # The larger the more complex
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0), # The larger the more complex
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # The larger the more complex
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 50, log=True), # The larger the more conservative
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True), # The larger the more conservative
            'gamma': trial.suggest_float('gamma', 0.01, 5.0, log=True), # The larger the more conservative
        }

        num_boost_round = 10000
        params.update(base_params)
        metric = base_params['eval_metric']
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'valid-{metric}')
        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        early_stopping_rounds=50,
                        verbose_eval=0,
                        callbacks=[pruning_callback])
        trial.set_user_attr('best_iteration', model.best_iteration)
        return model.best_score
    
    # Cast object features to categorical
    # purporsely not implemented here
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=50)

    best_round = study.best_trial.user_attrs["best_iteration"]
    best_params = study.best_trial.params

    return best_params, best_round



def train_model(df_all, df_train, df_valid, df_trainvalid, covariates, target, base_params=None):

    monotone_constraints = {
        'temperature': 1,
        'wind_speed_kmh': 1,
        'relative_humidity': -1,
        'slope_degrees': 1,
        }


    if base_params is None:
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            "random_state": 42,
            "monotone_constraints": {key: monotone_constraints[key] for key in covariates if key in monotone_constraints},
            'learning_rate': 0.02,
        }

    # Prepare arrays. Let's take a copy to be on the safe side
    X_all   = df_all[covariates].copy()
    X_train = df_train[covariates].copy()
    X_valid = df_valid[covariates].copy() if not df_valid.empty else None
    X_trainvalid = df_trainvalid[covariates].copy()

    y_all   = df_all[target].copy()
    y_train = df_train[target].copy()
    y_valid = df_valid[target].copy() if not df_valid.empty else None
    y_trainvalid = df_trainvalid[target].copy()


    # First stage: hyperparameter tuning. Each round with early stopping
    logger.info("stage 1: tuning model.")

    best_params, best_round = xgb_tune_fit(
        X_train=X_train, 
        y_train=y_train, 
        X_valid=X_valid, 
        y_valid=y_valid,
        base_params=base_params
        )


    # Second stage: retrain on train+valid dataset
    logger.info("stage 2: retrain on train+valid dataset")
    dtrainvalid = xgb.DMatrix(X_trainvalid, label=y_trainvalid, enable_categorical=True)

    best_params.update(base_params)
    best_params['learning_rate'] = 0.1 # hardcoded

    model0 = xgb.train(
        params=best_params,
        dtrain=dtrainvalid,
        evals=[(dtrainvalid, 'train+valid')],
        num_boost_round=best_round,
        verbose_eval=best_round
    )

    
    # Third stage: retrain on the whole dataset
    logger.info("stage 3: retrain on the whole dataset")
    dall = xgb.DMatrix(X_all, label=y_all, enable_categorical=True)

    model = xgb.train(
        params=best_params,
        dtrain=dall,
        evals=[(dall, 'all')],
        num_boost_round=best_round,
        verbose_eval=best_round
    )

    return model0, model


def get_predictive_intervals(model, df, covariates, pi_covariates, target,  alpha, heteroscedastic=False):

    if heteroscedastic:
        # Use the model to predict the mean and variance
        Warning("Heteroscedastic predictive intervals are not yet implemented.")

    residuals = df[target] - model.predict(xgb.DMatrix(df[covariates], enable_categorical=True))

    # Compute the quantiles for the predictive intervals
    if not pi_covariates:
        lo = residuals.quantile(alpha)
        hi = residuals.quantile(1 - alpha)
    else:
        tmp = df[covariates].copy()
        tmp["residuals"] = residuals
        lo = tmp.groupby(pi_covariates, observed=False)["residuals"].quantile(alpha).rename("lo")
        hi = tmp.groupby(pi_covariates, observed=False)["residuals"].quantile(1 - alpha).rename("hi")
   
    return {
        "lo": lo.to_dict(),
        "hi": hi.to_dict()
    }






def train_pipeline(
       df: pd.DataFrame,
       factor1: str,
       factor2: str,
       target: str,
       covariates: list,
       pi_covariates: list,
       alpha: float = 0.1,
):
    
    out = df.copy()
    point_prediction_dict = {}
    prediction_interval_dict = {}

    for g, df_all in out.groupby([factor1, factor2], observed=True, dropna=False):

        # I prefer not to silently drop missing values here
        usecols = [target] + covariates
        assert df_all[usecols].notnull().all().all(), f"Group {g} has missing values."

        # train-validation split
        df_train = df_all[df_all["split3"] == "train"]
        df_valid = df_all[df_all["split3"] == "valid"]
        df_trainvalid = df_all[df_all["split2"] == "train+valid"]

        # Check if train set is empty
        if df_train.empty:
            logger.warning(f"Skipping group {g} (no train rows).")
            continue

        # If y has only one unique value, skip
        if df_train[target].nunique() < 2:
            logger.warning(f"Skipping group {g} (only one unique target value).")
            continue
        
        # train the point prediction model
        model0, model = train_model(df_all, df_train, df_valid, df_trainvalid, covariates, target, base_params=None)

        # Model0 point predictions
        dall = xgb.DMatrix(df_all[covariates], label=df_all[target], enable_categorical=True)
        preds = model0.predict(dall)
        out.loc[df_all.index, "prediction_xgb0"] = preds

        # Model point predictions
        preds = model.predict(dall)
        out.loc[df_all.index, "prediction_xgb"] = preds
        point_prediction_dict[g] = model # final model is stored

        # get prediction intervals
        pi0 = get_predictive_intervals(model0, df_valid, covariates, pi_covariates, target, alpha=alpha,  heteroscedastic=False)
        pi = get_predictive_intervals(model, df, covariates, pi_covariates, target, alpha=alpha,  heteroscedastic=False)
        prediction_interval_dict[g] = pi # final model is stored

        # map prediction intervals to df
        idx = df_all.index

        if not pi_covariates:
            out.loc[idx, 'lo_xgb0'] = pi0['lo']
            out.loc[idx, 'hi_xgb0'] = pi0['hi']
        else:
            key_series = out.loc[idx, pi_covariates].agg(tuple, axis=1)
            out.loc[idx, 'lo_xgb0'] = key_series.map(pi0['lo'])
            out.loc[idx, 'hi_xgb0'] = key_series.map(pi0['hi'])

        if not pi_covariates:
            out.loc[idx, 'lo_xgb'] = pi['lo']
            out.loc[idx, 'hi_xgb'] = pi['hi']
        else:
            key_series = out.loc[idx, pi_covariates].agg(tuple, axis=1)
            out.loc[idx, 'lo_xgb'] = key_series.map(pi['lo'])
            out .loc[idx, 'hi_xgb'] = key_series.map(pi['hi'])

    return out, point_prediction_dict, prediction_interval_dict


def add_base_model_predictions_to_df(
        df: pd.DataFrame, 
        factor1: str, 
        factor2: str, 
        target: str, 
        alpha: float = 0.1):
    
    # --- Call args (debug) ---
    logger.debug("called with df.shape=%s, factor1=%s, factor2=%s, target=%s, alpha=%s",
        df.shape, factor1, factor2, target, alpha)
    
    # create a copy
    out = df.copy()

    # Point Predictions
    factors = [factor1, factor2]
    grp_means = df.groupby(factors, observed=False)[target].mean()
    out['prediction_base'] = out.set_index(factors).index.map(grp_means)

    # Compute residuals
    out['residual_base'] = out[target] - out['prediction_base']

    # Predictive Intervals
    grp_quantiles = out.groupby(factors, observed=False)['residual_base'].quantile([alpha, 1-alpha]).unstack()
    out['lo_base'] = out.set_index(factors).index.map(grp_quantiles[alpha])
    out['hi_base'] = out.set_index(factors).index.map(grp_quantiles[1-alpha])

    # drop residuals
    out = out.drop(columns=['residual_base'])

    logger.info("Added %s, %s, %s, %s, %s", 'prediction_base', 'lo_base', 'hi_base', 'residual_base', 'grp_quantiles')

    return out


# ------ saving artifacts ------


def _save_model(model, pi, meta, model_path, pi_path, meta_path):

    joblib.dump(model, model_path)

    pi_serialized = serialize(pi)

    with open(pi_path, "w", encoding="utf-8") as f:
        json.dump(pi_serialized, f, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.debug("Saved model -> %s", model_path)
    logger.debug("Saved PI    -> %s", pi_path)
    logger.debug("Saved meta  -> %s", meta_path)

def dump_artifacts(
        artifacts_dir,
        df,
        pp_dict,
        pi_dict,
        factor1,
        factor2,
        covariates,
        pi_covariates,
        target,
        alpha,
        ratio
):
    for g, df_all in df.groupby([factor1, factor2], observed=True, dropna=False):

        if g not in pp_dict:
            logging.warning(f"Group {g} not in pp_dict, skipping...")
            continue

        model = pp_dict[g]
        pi = pi_dict[g]

        model_path    = f"{artifacts_dir}/model_{g[0]}_{g[1]}.pkl"
        pi_path       = f"{artifacts_dir}/pi_{g[0]}_{g[1]}.json"
        meta_path     = f"{artifacts_dir}/meta_{g[0]}_{g[1]}.json"

        meta = {
            "n_train": int(len(df_all)),
            "covariates": list(df_all.columns),
            "covariates": covariates,
            "pi_covariates": pi_covariates,
            "target": target,
            "alpha": alpha,
            "ratio": ratio
        }

        _save_model(model, pi, meta, model_path, pi_path, meta_path)


# ---- to be used by the eavluation_model script ----

def eval_point_prediction(X_new, model, covariates, mgr):
    
    # 1) Build ordered, typed row
    X_new = X_new[covariates].copy()
    X_ordered = mgr.apply(X_new[covariates], strict=True, include=covariates).copy()

    # 2) Predict with TreeSHAP
    dnew = xgb.DMatrix(X_ordered, enable_categorical=True, feature_names=covariates)
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


def eval_prediction_interval(X_new, model, covariates):

    assert isinstance(model, dict) and "lo" in model and "hi" in model, \
        "PI model must be a dictionary with keys 'lo' and 'hi'"

    assert all(c in X_new.columns for c in covariates), "All columns in covariates must be present in X_new"
    
    return {
        "lo": model["lo"].get(tuple(X_new[covariates].values[0]), None) if covariates else model["lo"],
        "hi": model["hi"].get(tuple(X_new[covariates].values[0]), None) if covariates else model["hi"]
    }


def elliptical_propagation_speed(circular_speed, lo_circular, hi_circular, ratio=3.0):

    major_axis_speed = np.sqrt(ratio)*circular_speed
    lo_major = np.sqrt(ratio)*lo_circular
    hi_major = np.sqrt(ratio)*hi_circular
    lo_major = np.minimum(major_axis_speed, lo_major) # speed cannot be negative

    minor_axis_speed = circular_speed/np.sqrt(ratio)
    lo_minor = lo_major/ratio
    hi_minor = hi_major/ratio
    lo_minor = np.minimum(minor_axis_speed, lo_minor) # speed cannot be negative

    #return major_axis_speed, lo_major, hi_major, minor_axis_speed, lo_minor, hi_minor

    return {
        "elliptical_speed": major_axis_speed,
        "lo_elliptical": lo_major,
        "hi_elliptical": hi_major,
        "minor_axis_speed": minor_axis_speed,
        "lo_minor": lo_minor,
        "hi_minor": hi_minor
    }


def add_elliptical_propagation_speed_to_df(
    df: pd.DataFrame,
    circular_col: str = 'prediction_xgb',
    lo_circular_col: str = "lo_xgb",
    hi_circular_col: str = "hi_xgb",
    ratio: float = 3.0
) -> pd.DataFrame:
    """
    Compute elliptical propagation speed and add it to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing circular propagation speed.
        circular_col (str): Column name for circular propagation speed (m/min).
        lo_circular_col (str): Column name for lower bound of circular speed.
        hi_circular_col (str): Column name for upper bound of circular speed.
        ratio (float): Ratio of major to minor axis speeds.
    Returns:
        pd.DataFrame: DataFrame with added elliptical propagation speed columns.
    """

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, circular_col=%s, lo_circular_col=%s, hi_circular_col=%s, ratio=%s",
        df.shape, circular_col, lo_circular_col, hi_circular_col, ratio
    )

    # validate
    for col in [circular_col, lo_circular_col, hi_circular_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df.")

    # create a copy
    out = df.copy()
    columns_before = out.columns.tolist()
    out = out.join(out.apply(lambda row: pd.Series(elliptical_propagation_speed(row[circular_col], row[lo_circular_col], row[hi_circular_col], ratio=ratio)), axis=1))
    new_cols = [c for c in out.columns if c not in columns_before]
    logger.info("Added %s", new_cols)
    return out