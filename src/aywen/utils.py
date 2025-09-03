from functools import wraps
import datetime as dt
import pandas as pd
import logging
import os
import numpy as np
import geopandas as gpd


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
        
def get_region(shapefile, coords):
    """
    Get the region of a point from a shapefile
    
    Parameters
    ----------
    shapefile : str
        Path to the shapefile
    coords : tuple
        (x, y) coordinates of the point
        
    Returns
    -------
    region : str
        The region of the point
    """
    
    # read shapefile
    gdf = gpd.read_file(shapefile)
    