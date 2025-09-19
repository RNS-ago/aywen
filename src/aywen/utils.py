from functools import wraps
import datetime as dt
import pandas as pd
import logging
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



# --- Configure logging globally ---
logging.basicConfig(
    level=logging.INFO,  # Default level (can be changed later)
    format="%(message)s"  # Only show the message (no timestamps/levels unless you want them)
)


def concatnate_name_from_paths(paths: list[str]) -> str:
    report_type = os.path.splitext(os.path.basename(paths[0]))[0].split('_')[0]
    
    report_date_range = []
    for path in paths:
        report_date_range.append(os.path.splitext(os.path.basename(path))[0].split('_')[1])
        
    report_date_range = '_'.join(report_date_range)
    
    return f"{report_type}_{report_date_range}"
    
    
def assert_time_diff(df1, df2, key1, key2, type='equal'):

    left = df1[['fire_id', key1]].copy()
    right = df2[['fire_id', key2]].copy()

    # cast to datetime
    left[key1] = pd.to_datetime(left[key1])
    right[key2] = pd.to_datetime(right[key2])

    # group by fire_id and get min
    right = right.groupby('fire_id').min().reset_index()
    #  merge
    merged = pd.merge(left, right, on='fire_id', how='left')

    # error
    merged['time_diff'] = (merged[key1] - merged[key2]).dt.total_seconds() / 60
    merged.dropna(subset=['time_diff'], inplace=True)

    if type == 'equal':
        assert np.all(merged['time_diff'] == 0), "Error: There are records with non-zero time difference"
    elif type == 'positive':
        assert np.all(merged['time_diff'] >= 0), "Error: There are records with non-positive time difference"
    elif type == 'negative':
        assert np.all(merged['time_diff'] <= 0), "Error: There are records with non-negative time difference"
        




class SpeedCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to categorize propagation speed into ordinal bins,
    with optional visualization support.
    """
    def __init__(self, 
                 bins=None, 
                 labels=None, 
                 right=False):
        self.bins = bins if bins is not None else [0, 1.7, 10, 33, 83, np.inf] # meters / minute
        self.labels = labels if labels is not None else ['baja', 'media', 'alta', 'muy alta', 'extrema']
        self.right = right
   
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()    
        X_cat = pd.cut(
            X,
            bins=self.bins,
            labels=self.labels,
            right=self.right
        )
        return X_cat


class TimeOfDayFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, period=24):
        """
        Parameters:
        - period: periodicity of the cycle (default: 24 for hours)
        """
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform input hours into sine and cosine components with phase shift.
        """

        X = np.asarray(X).reshape(-1)
        theta = 2 * np.pi * X / self.period

        cos_hour = np.cos(theta)
        sin_hour = np.sin(theta)

        return sin_hour, cos_hour
    
    
class MonthCycleFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, period=12, peak_month=1):
        """
        period: number of months in a year (default 12)
        peak_month: the month (1–12) where the sine cycle should peak
        """
        self.period = period
        self.phase = 2 * np.pi * (peak_month - 1) / period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: array-like of months [1–12]
        Returns: DataFrame with sin_month and cos_month
        """
        X = np.asarray(X).reshape(-1)
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Input must be integer months in [1, 12].")
        theta = 2 * np.pi * (X - 1) / self.period
        
        return np.sin(theta - self.phase), np.cos(theta - self.phase)


def make_dtype_dict(df: pd.DataFrame):
    dtype_dict = {}
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            cat = df[col].dtype
            # keep categories and order
            dtype_dict[col] = pd.api.types.CategoricalDtype(
                categories=cat.categories,
                ordered=cat.ordered
            )
        else:
            dtype_dict[col] = df[col].dtype
    return dtype_dict
