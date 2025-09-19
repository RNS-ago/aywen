import pandas as pd
import logging

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

# ------- postprocessing pipeline -------

def postprocessing_pipeline(df, thresholds=None):

    if thresholds is None:
        thresholds = {
            "dt_minutes": {"lower": 10, "upper": 30},
            "initial_radius_m": {"lower": 2, "upper": 100}
        }

    out = df.copy()
    out = drop_missing_rows(out)
    out = row_filter(out, thresholds)

    return out
