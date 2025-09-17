import pandas as pd
import logging

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
