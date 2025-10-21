import logging
import pandas as pd
from pandas.testing import assert_frame_equal
from aywen.utils import DtypeManager
import xgboost as xgb
import numpy as np


logger = logging.getLogger("aywen_logger")


def assert_df(
        new_df: pd.DataFrame, 
        old_df: pd.DataFrame,
        time_cols: list = [],
        numeric_cols: list = [],
        exclude_cols: list = []
    ) -> None:
    left = new_df.copy()
    right = old_df.copy()
    
    

    # common columns
    len_left = len(left.columns)
    len_right = len(right.columns)
    common_cols = list(set(left.columns).intersection(right.columns) - set(exclude_cols))
    logger.info("cols new=%s, old=%s, common=%s", len_left, len_right, len(common_cols))
    logger.info("Common columns: %s", common_cols)
    left = left[common_cols].reset_index(drop=True)
    right = right[common_cols].reset_index(drop=True)


    # fixing timestamps columns
    for col in time_cols:
        if col in right.columns:
            left[col] = pd.to_datetime(left[col], errors='coerce')
            right[col] = pd.to_datetime(right[col], errors='coerce')

    # fixing numeric columns
    for col in numeric_cols:
        if col in right.columns:
            left[col] = pd.to_numeric(left[col], errors='coerce')
            right[col] = pd.to_numeric(right[col], errors='coerce')

    assert_frame_equal(left, right, check_dtype=False, check_like=True)


def assert_df_from_file(
        df : pd.DataFrame,
        filename : str,
        time_cols: list = [],
        numeric_cols: list = [],
        exclude_cols: list = []
    ) -> None:
    
    # Read old dataframe
    old_df = pd.read_csv(filename, low_memory=False)
    # Ensure factor2 is clean integer-like strings ("1"..."5")
    factor2 = 'zone_NS'
    if factor2 in old_df.columns and factor2 in df.columns:
        old_df[factor2] = pd.to_numeric(old_df[factor2], errors='coerce').astype('Int64').astype(str)
    # dtype management
    mgr = DtypeManager.from_df(df)
    include = [col for col, dt in mgr.to_dict().items() if isinstance(dt, pd.CategoricalDtype)]
    old_df = mgr.apply(old_df, drop_extras=False, include=include)

    # Call assert_df
    assert_df(df, old_df, time_cols=time_cols, numeric_cols=numeric_cols, exclude_cols=exclude_cols)



def assert_predictions_match(df, pp_dict, covariates, factor1, factor2):
    
    # Load schema
    mgr = DtypeManager.from_df(df[covariates])

    # Model point predictions
    random_row = df.sample(n=1, random_state=42)
    idx = random_row.index[0]
    logger.info('Selected row index = %s', idx)

    # Get factors and model
    f1 = random_row[factor1].values[0]
    f2 = random_row[factor2].values[0]
    model = pp_dict[(f1, f2)]
    pred_random = random_row['prediction_xgb'].iloc[0]

    # Build df_new from the same row, enforcing the same order & copy
    X_random = random_row[covariates].iloc[0].to_dict()
    df_new = pd.DataFrame([X_random], columns=covariates).copy()
    df_new = mgr.apply(df_new,  strict=False)
    dnew = xgb.DMatrix(df_new, enable_categorical=True)
    pred_xgb = model.predict(dnew)[0]

     # predict with eval_point_prediction
    result = eval_point_prediction(df_new, model, covariates, mgr)
    pred_eval = result['prediction']

    logger.info("Prediction at random row = %s", pred_random)
    logger.info("Prediction with xgboost = %s", pred_xgb)
    logger.info("Prediction with eval_point_prediction = %s", pred_eval)

    assert pred_random == pred_xgb, "Predictions random vs xgboost do not match!"
    assert np.isclose(pred_random, pred_eval), "Predictions random vs eval_point_prediction do not match!"

