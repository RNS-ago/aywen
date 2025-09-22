import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from aywen.fire_features import DEFAULT_COLUMNS

logger = logging.getLogger("aywen_logger")

def row_filter(df, thresholds):
    """
    Filter DataFrame rows based on upper and lower thresholds for one or more columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    thresholds : dict
        A dictionary where keys are column names and values are dictionaries with optional 
        'lower' and/or 'upper' keys specifying the thresholds.
        Example:
            {
                "col1": {"lower": 10, "upper": 100},
                "col2": {"upper": 50},
                "col3": {"lower": 0}
            }

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only rows within the given thresholds.
    """
    out = df.copy()
    mask = pd.Series(True, index=out.index)  # start with all rows

    for col, bounds in thresholds.items():
        if "lower" in bounds:
            mask &= out[col] >= bounds["lower"]
        if "upper" in bounds:
            mask &= out[col] <= bounds["upper"]
        logger.info("Filtering %s with bounds %s. Remaining rows: %d", col, bounds, mask.sum())

    return out[mask]

def drop_missing_rows(df, subset=None):
    """
    Drop rows with any missing values in the specified subset of columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    subset : list, optional
        List of column names to check for missing values. If None, all columns are checked.

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows containing missing values in the specified columns removed.
    """
    initial_count = len(df)
    out = df.dropna(subset=subset)
    final_count = len(out)
    logger.info("Dropped %d rows with missing values. Remaining rows: %d", initial_count - final_count, final_count)
    return out

# ----------- train-validate-test split -------



def train_test_split_per_zone(df, zone_col="zone_alert", test_frac=0.20, val_frac=0.20, random_state=42):
    """
    Returns df_train, df_val, df_test with disjoint indices.
    Splits are performed independently within each zone.
    """
    parts = []
    for z, dfz in df.groupby(zone_col, observed=False):
        # test split within this zone
        df_temp, df_test = train_test_split(
            dfz, test_size=test_frac, random_state=random_state, shuffle=True
        )
        # validation split within the remaining data
        val_size_rel = val_frac / (1.0 - test_frac)  # so overall val ~= val_frac
        df_train, df_val = train_test_split(
            df_temp, test_size=val_size_rel, random_state=random_state, shuffle=True
        )

        df_train = df_train.copy(); df_train["dataset"] = "train"
        df_val   = df_val.copy();   df_val["dataset"]   = "val"
        df_test  = df_test.copy();  df_test["dataset"]  = "test"

        parts.append((df_train, df_val, df_test))

    # concat all zones back
    df_train = pd.concat([p[0] for p in parts]).sort_index()
    df_val   = pd.concat([p[1] for p in parts]).sort_index()
    df_test  = pd.concat([p[2] for p in parts]).sort_index()

    logger.debug("Train=%s, Val=%s, Test=%s", len(df_train), len(df_val), len(df_test))
    return df_train, df_val, df_test

def add_train_test_split_to_df(
        df: pd.DataFrame,
        zone_col: str = "zone_alert",
        factor1: str = "zone_WE",
        factor2: str = "zone_NS",
        split2_col : str = "split2",
        split3_col : str = "split3"
) -> pd.DataFrame:
    
    # --- Call args (debug) ---
    logger.debug("called with df.shape=%s, zone_col=%s, split2_col=%s, split3_col=%s",
        df.shape, zone_col, split2_col, split3_col
    )

    # create a copy
    out = df.copy()

    # splitting per zone
    df_train, df_val, df_test = train_test_split_per_zone(df, test_frac=0.20, val_frac=0.20, random_state=42)

    # --- split flags ---
    out["split3"] = "train"
    out.loc[out.index.isin(df_val.index), "split3"] = "valid"
    out.loc[out.index.isin(df_test.index), "split3"] = "test"

    out['split2'] = "train+valid"
    out.loc[out.index.isin(df_test.index), "split2"] = "test"

    # Optional: drop NaNs to avoid spurious rows
    df_chk = out.dropna(subset=['initial_fuel_reduced'])

    counts = (df_chk
        .groupby([factor1, factor2, 'initial_fuel_reduced', 'split3'], observed=True)
        .size()
        .unstack('split3', fill_value=0)
    )

    # ensure train column exists even if there are no train rows at all
    if 'train' not in counts.columns:
        counts['train'] = 0

    # rows present somewhere (they all are) but missing in train
    missing = counts[counts['train'] == 0]

    assert missing.empty, (
        "Some initial_fuel_reduced levels are missing from the train split:\n"
        + missing.reset_index()[[factor1, factor2, 'initial_fuel_reduced']].to_string(index=False)
    )

    logger.info("Added %s and %s.", split2_col, split3_col)

    return out


# ------- postprocessing pipeline -------

def postprocessing_pipeline(df, thresholds=None, zone_col="zone_alert"):

    if thresholds is None:
        thresholds = {
            "dt_minutes": {"lower": 10, "upper": 30},
            "initial_radius_m": {"lower": 2, "upper": 100}
        }

    out = df.copy()
    out = drop_missing_rows(out)
    out = row_filter(out, thresholds)
    out = add_train_test_split_to_df(out, zone_col=zone_col)

    return out
