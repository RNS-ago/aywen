import logging
import pandas as pd
import numpy as np
import geopandas as gpd
import os


logger = logging.getLogger("aywen_logger")


def get_zone(
    shapefile_path: str,
    crs: str = "EPSG:4326",
    coords: tuple | list[tuple] = None,
    df: pd.DataFrame = None,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    zone_col: str = "area",
    new_zone_col: str = "region",
    ) -> str | gpd.GeoDataFrame:
    """
    Get the region of coordinates from a shapefile, returning a GeoDataFrame if a DataFrame is provided.
    
    Parameters:
        shapefile (str): Path to the shapefile.
        crs (str): Coordinate reference system. Default is "EPSG:4326".
        coords (tuple or list[tuple]): Coordinates to get the region for. If a tuple is provided, it should be (longitude, latitude). If a list of tuples is provided, it should be a list of (longitude, latitude) tuples.
        df (pd.DataFrame): DataFrame containing the coordinates. If provided, the function will return a GeoDataFrame instead of a list of dictionaries.
        lon_col (str): Name of the longitude column in the DataFrame. Default is "longitude".
        lat_col (str): Name of the latitude column in the DataFrame. Default is "latitude".
        zone_col (str): Name of the column containing the zones in the shapefile. Default is "area".
        new_zone_col (str): Name of the column to add to the DataFrame. Default is "region".
    
    Returns:
        dict or list[dict] or gpd.GeoDataFrame: If a DataFrame is provided, returns a GeoDataFrame with the coordinates and the region. If a tuple is provided, returns a list of dictionaries with the coordinates and the region.
    """
    
    try:
        if isinstance(shapefile_path, str) == False:
            raise TypeError("shapefile_path must be a string.")
        if shapefile_path.endswith(".shp") == False:
            raise ValueError("shapefile_path must be a .shp file.")
        if os.path.exists(shapefile_path) == False:
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        if isinstance(crs, str) == False:
            raise TypeError("crs must be a string.")
        if isinstance(coords, tuple) == False and isinstance(coords, list) == False:
            raise TypeError("coords must be a tuple or a list of tuples.")
        if isinstance(df, pd.DataFrame) == False and df is not None:
            raise TypeError("df must be a pandas DataFrame or None.")
        if isinstance(lon_col, str) == False:
            raise TypeError("lon_col must be a string.")
        if isinstance(lat_col, str) == False:
            raise TypeError("lat_col must be a string.")
        if isinstance(zone_col, str) == False:
            raise TypeError("zone_col must be a string.")
        if isinstance(new_zone_col, str) == False:
            raise TypeError("new_zone_col must be a string.")
    except Exception:
        logger.exception("Invalid input to get_zone.")
        raise
            

    shape_df = gpd.read_file(shapefile_path)
    shape_df = shape_df.set_crs(epsg=4326)


    if df is not None:
        x_coords = df[lon_col]
        y_coords = df[lat_col]
        coords_df = df.copy()
    else:
        if isinstance(coords, tuple):
            coords = [coords]

        coords = np.array(coords)
        coords = np.transpose(coords)
        x_coords = coords[0]
        y_coords = coords[1]

        coords_df = pd.DataFrame(np.array([x_coords, y_coords]).T, columns=[lon_col, lat_col])


    pts = gpd.GeoDataFrame(
        data=coords_df,
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=crs,
    )

    gdf = gpd.sjoin(
        left_df=pts,
        right_df=shape_df[[zone_col, "geometry"]],
        how="left",
        predicate="within",
    )
    
    gdf = gdf.dropna(subset=[zone_col])
    gdf = gdf.rename(columns={zone_col: new_zone_col})
    gdf = gdf.drop(columns=["index_right"])


    if gdf.empty:
        logger.debug("None of the coordintes were in the regions in the shapefile")
        return None
    
    if df is not None:
        return gdf
    else:
        return [{lat_col:row[1][lat_col], lon_col:row[1][lon_col], new_zone_col:row[1][new_zone_col]} for row in gdf.iterrows()]


def load_preprocessed_data(fire_data_path: str, dispatch_data_path: str):
    
    fire_df = pd.read_csv(fire_data_path)
    dispatch_df = pd.read_csv(dispatch_data_path)
    
    fire_datetime_cols = [col for col in fire_df.columns if 'datetime' in col]
    dispatch_datetime_cols = [col for col in dispatch_df.columns if col.startswith('hr_') or 'datetime' in col]
    
    for col in fire_datetime_cols:
        fire_df[col] = pd.to_datetime(fire_df[col])
        
    for col in dispatch_datetime_cols:
        dispatch_df[col] = pd.to_datetime(dispatch_df[col])
        
    return fire_df, dispatch_df


def classify_response_type(dispatch_df):
    glosas = dispatch_df['glosa'].unique()
    
    mapping = {}
    
    for glosa in glosas:
        if 'BHA' in glosa:
            mapping[glosa] = 'H'
        elif 'AA' in glosa:
            mapping[glosa] = 'A'
        elif 'BA' in glosa:
            mapping[glosa] = 'T'
        else:
            mapping[glosa] = 'OTRO'
            
    dispatch_df['response_type'] = dispatch_df['glosa'].map(mapping)
    
    dispatch_df = dispatch_df[dispatch_df['response_type'] != 'OTRO']
    
    return dispatch_df

def print_missing_values(df):
    
    print("Number of missing values:")
    cols = [key for key in df.columns if "hr" in key]
    print(df[cols+["recurso"]].isnull().sum())

def filter_dispatchs_by_first_response(dispatch_df):
    try:
        if len(dispatch_df[dispatch_df["start_datetime"] > dispatch_df["hr_arribo"]]) != 0:
            raise ValueError("There are records with start_datetime > hr_arribo")
    except Exception:
        logger.exception("Failed to filter dispatchs by first response")
        raise
    
    filtered_dispatch_df = dispatch_df[dispatch_df['recurso'].isin(['H', 'T'])]
    
    return filtered_dispatch_df

def add_first_dispatch_arrival(fire_df, dispatch_df):
    """
    Add the arrival time of the first dispatch to the fire dataframe.
    
    Parameters:
        fire_df (pd.DataFrame): The DataFrame containing the fire data.
        dispatch_df (pd.DataFrame): The DataFrame containing the dispatch data.
    Returns:
        pd.DataFrame: The updated fire DataFrame.
    """
    try:
        if (dispatch_df['start_datetime'] > dispatch_df['hr_arribo']).any():
            raise ValueError("There are records with start_datetime > hr_arribo")
    except Exception:
        logger.exception("Failed to add first dispatch arrival")
        raise
    
    
    right = dispatch_df.groupby('fire_id')['hr_arribo'].min().reset_index()
    right = right.rename(columns={'hr_arribo': 'arrival_datetime_des'})
    
    if 'arrival_datetime_des' in fire_df.columns:
        fire_df = fire_df.drop(columns=['arrival_datetime_des'])
        
    fire_df = pd.merge(fire_df, right, on='fire_id', how='left')
    
    try:
        key1 = 'arrival_datetime_inc' # arrival time from incendios
        key2 = 'arrival_datetime_des' # arrival time from despachos
        key3 = 'start_datetime' # start time from despachos
        s12 = ((fire_df[key1] - fire_df[key2]).dt.total_seconds() / 60).dropna()
        s31 = ((fire_df[key3] - fire_df[key1]).dt.total_seconds() / 60).dropna()
        s32 = ((fire_df[key3] - fire_df[key2]).dt.total_seconds() / 60).dropna()
        if np.any(s12 > 0):
            raise ValueError("There are records with arrival_datetime_inc > arrival_datetime_des")
        if np.any(s31 > 0):
            raise ValueError("There are records with start_datetime > arrival_datetime_inc")
        if np.any(s32 > 0):
            raise ValueError("There are records with start_datetime > arrival_datetime_des")
    except Exception:
        logger.exception("Failed to add first dispatch arrival")
        raise
    return fire_df

def check_for_outlier_fires(fire_df, outlier_tolerance=50000):
    """
    Check for outlier fires based on the difference between the start time of the fire and the arrival time of the first response.
    
    Parameters:
        fire_df (pd.DataFrame): The DataFrame containing the fire data.
        outlier_tolerance (int): The tolerance for identifying outlier fires (in minutes).
    Returns:
        list[str]: A list of outlier fire IDs.
    """
    
    key1 = 'arrival_datetime_inc' # arrival time from incendios
    key2 = 'arrival_datetime_des' # arrival time from despachos
    key3 = 'start_datetime' # start time from despachos
    
    s13 = ((fire_df[key1] - fire_df[key3]).dt.total_seconds() / 60).dropna()
    s23 = ((fire_df[key2] - fire_df[key3]).dt.total_seconds() / 60).dropna()
    
    outlier_fire_ids = fire_df.loc[s13 > outlier_tolerance, 'fire_id'].unique()
    
    if len(outlier_fire_ids) == 0:
        logger.info("There are no outlier fires")
    else:
        logger.info("There are %s outlier fires at %s", len(outlier_fire_ids), outlier_fire_ids)
    
    return outlier_fire_ids

def filter_common_fire_ids(fire_df, dispatch_df):
    """
    Filter the fire and dispatch dataframes to only include common fire IDs.
    
    Parameters:
        fire_df (pd.DataFrame): The DataFrame containing the fire data.
        dispatch_df (pd.DataFrame): The DataFrame containing the dispatch data.
    Returns:
        pd.DataFrame: The filtered fire DataFrame.
        pd.DataFrame: The filtered dispatch DataFrame. 
    """
    common_ids = set(fire_df['fire_id']).intersection(set(dispatch_df['fire_id']))
    
    if len(common_ids) == 0:
        logger.info("There are no common fire IDs")
        
    fire_df = fire_df[fire_df['fire_id'].isin(common_ids)]
    dispatch_df = dispatch_df[dispatch_df['fire_id'].isin(common_ids)]
    
    logger.info("There are %s fires with common IDs", len(fire_df))
    logger.info("There are %s dispatches with common IDs", len(dispatch_df))
    
    return fire_df, dispatch_df
