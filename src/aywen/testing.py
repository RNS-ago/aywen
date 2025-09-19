import logging
import pandas as pd
from pandas.testing import assert_frame_equal
from aywen.utils import make_dtype_dict


logger = logging.getLogger("aywen_logger")


def test_df(
        new_df: pd.DataFrame, 
        old_df: pd.DataFrame,
        time_cols: list = [],
        exclude_cols: set = set()
    ) -> None:

    left = new_df.copy()
    right = old_df.copy()

    # common columns
    len_left = len(left.columns)
    len_right = len(right.columns)
    common_cols = sorted(set(left.columns).intersection(right.columns) - exclude_cols)
    logger.info("cols new=%s, old=%s, common=%s", len_left, len_right, len(common_cols))
    left = left[common_cols].sort_index(axis=1)
    right = right[common_cols].sort_index(axis=1)

    # common rows based on fire_id
    len_left = len(left)
    len_right = len(right)
    common_ids = set(left["fire_id"]).intersection(right["fire_id"])
    left = left[left["fire_id"].isin(common_ids)].reset_index(drop=True)
    right = right[right["fire_id"].isin(common_ids)].reset_index(drop=True)
    logger.info("rows: new=%s, old=%s, common=%s", len_left, len_right, len(common_ids))

    # fixing timestamps columns
    for col in time_cols:
        if col in right.columns:
            left[col] = pd.to_datetime(left[col], errors='coerce')
            right[col] = pd.to_datetime(right[col], errors='coerce')

    assert_frame_equal(left, right, check_dtype=False, check_like=True)


def test_df_from_file(
        df : pd.DataFrame,
        filename : str,
        time_cols: list = [],
        exclude_cols: set = set()
    ) -> None:

    # Read old dataframe with correct dtypes
    dtype_dict = make_dtype_dict(df)
    #new_cols = list(dtype_dict.keys())
    
    read_cols = pd.read_csv(filename, nrows=0).columns.tolist()
    # use_cols = [col for col in new_cols if col in old_cols]
    dtype = {col: dtype_dict[col] for col in read_cols if col in dtype_dict and dtype_dict[col] == 'category'}

    # Read old dataframe
    old_df = pd.read_csv(filename, dtype=dtype)

    # Call test_df
    test_df(df, old_df, time_cols=time_cols, exclude_cols=exclude_cols)

