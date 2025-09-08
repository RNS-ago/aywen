import logging
import pandas as pd
import numpy as np
import geopandas as gpd
import os


logger = logging.getLogger("aywen_logger")


def add_zones_to_df(df,
    shapefile_path: str,
    crs: str = "EPSG:4326",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    zone_col: str = "area",
    new_zone_col: str = "zone_alert",
    ) -> pd.DataFrame:
    """
    Add the zone to a DataFrame containing the coordinates based on a shapefile.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the coordinates.
        shapefile_path (str): Path to the shapefile.
        crs (str): Coordinate reference system. Default is "EPSG:4326".
        lon_col (str): Name of the longitude column in the DataFrame. Default is "longitude".
        lat_col (str): Name of the latitude column in the DataFrame. Default is "latitude".
        zone_col (str): Name of the column containing the zones in the shapefile. Default is "area".
        new_zone_col (str): Name of the column to add to the DataFrame. Default is "region".
    
    Returns:
        gpd.GeoDataFrame: Returns a GeoDataFrame containing the original columns and a new column with the zone.
    """
    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, shapefile_path=%s, crs=%s, lon_col=%s, lat_col=%s, zone_col=%s, new_zone_col=%s", df.shape, shapefile_path, crs, lon_col, lat_col, zone_col, new_zone_col)
    
    
    #--- Validate input ---
    try:
        if isinstance(shapefile_path, str) == False:
            raise TypeError("shapefile_path must be a string.")
        if shapefile_path.endswith(".shp") == False:
            raise ValueError("shapefile_path must be a .shp file.")
        if os.path.exists(shapefile_path) == False:
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        if isinstance(crs, str) == False:
            raise TypeError("crs must be a string.")
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
        logger.exception("Invalid input to add_zone_to_df.")
        raise
    
    
    #--- Create GeoDataFrame ---
    logger.info("Creating GeoDataFrame from coordinates.")
    df = df.copy()

    try:
        shape_df = gpd.read_file(shapefile_path)
        shape_df = shape_df.set_crs(crs=crs)
    except Exception:
        logger.exception("Failed to read shapefile.")
        raise

    pts = gpd.GeoDataFrame(
        data=df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
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
    gdf = gdf.drop(columns=["index_right", "geometry"])


    if gdf.empty:
        logger.info("None of the coordintes were in the regions in the shapefile")
        return None
    else:
        logger.info("Successfully added zones to %s out of %s rows.", gdf.shape[0], df.shape[0])
        return gdf

def split_zones(
    df,
    zone_col: str = "zone_alert",
    separation_keys: list[str] = ["zone_WE", "zone_NS"],
) -> pd.DataFrame:
    """
    Split the zones in a DataFrame into separate sub-zone columns based on the separation_keys.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the zones.
        zone_col (str): The name of the column containing the zones.
        separation_keys (list[str]): The list of keys to split the zones into.
    Returns:
        pd.DataFrame: The DataFrame with the subdivided zones.
    """
    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, zone_col=%s, separation_keys=%s", df.shape, zone_col, separation_keys)
    
    
    #--- Validate input ---
    try:
        if isinstance(df, pd.DataFrame) == False:
            raise TypeError("df must be a pandas DataFrame.")
        if isinstance(zone_col, str) == False:
            raise TypeError("zone_col must be a string.")
        if isinstance(separation_keys, list) == False:
            raise TypeError("separation_keys must be a list.")
        if isinstance(separation_keys[0], str) == False:
            raise TypeError("separation_keys must be a list of strings.")
        if zone_col not in df.columns:
            raise ValueError(f"zone_col {zone_col} not in df.columns")
    except Exception:
        logger.exception("Invalid input to split_zones.")
        raise


    # --- Split zones ---
    logger.info("Splitting zones into separate columns.")
    df = df.copy()

    # One-liner split with automatic column naming
    split_cols = df[zone_col].astype(str).str.split(" ", expand=True)

    # Rename and assign only the columns we need
    logger.info("Assigning zones to separate columns.")
    for i, key in enumerate(separation_keys):
        if i < split_cols.shape[1]:
            # Create categorical with unique values as categories
            unique_vals = split_cols[i].dropna().unique()
            df[key] = pd.Categorical(
                split_cols[i], categories=unique_vals, ordered=True
            )

    
    logger.info("Successfully split %s column zones into %s columns.", zone_col, len(separation_keys))
    return df

def get_zone(coords: tuple | list[tuple],
    shapefile_path: str,
    crs: str = "EPSG:4326",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    zone_col: str = "area",
    new_zone_col: str = "zone_alert",
    ) -> dict | list[dict]:
    """
    Get the region of coordinates from a shapefile.
    
    Parameters:
        coords (tuple or list[tuple]): Coordinates to get the region for. If a tuple is provided, it should be (longitude, latitude). If a list of tuples is provided, it should be a list of (longitude, latitude) tuples.
        shapefile_path (str): Path to the shapefile.
        crs (str): Coordinate reference system. Default is "EPSG:4326".
        lon_col (str): Name of the longitude column in the DataFrame. Default is "longitude".
        lat_col (str): Name of the latitude column in the DataFrame. Default is "latitude".
        zone_col (str): Name of the column containing the zones in the shapefile. Default is "area".
        new_zone_col (str): Name of the column to add to the DataFrame. Default is "region".
    
    Returns:
        dict or list[dict]: Returns a dictionary or list of dictionaries with the coordinates and the region.
    """
    #--- Call args (debug) ---
    logger.debug("called with coords=%s, shapefile_path=%s, crs=%s, lon_col=%s, lat_col=%s, zone_col=%s, new_zone_col=%s", coords, shapefile_path, crs, lon_col, lat_col, zone_col, new_zone_col)   
    
    
    #--- Validate input ---
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
    
    
    #--- Convert coords to numpy array and validate ---
    try:
        if isinstance(coords, tuple):
            if len(coords) != 2:
                raise ValueError("coords must be a tuple of length 2 (longitude, latitude). Provided coords are of length %s.", len(coords))
            coords = [coords]
            
        coords = np.array(coords)
        if coords.shape[1] != 2:
            raise ValueError("coords must be a list of tuples of length 2 [(longitude, latitude),...]. Provided coords are of shape %s.", coords.shape)
    except Exception:
        logger.exception("Invalid coordinates shape.")
        raise
    
    
    # --- Create coordinates DataFrame and add zones ---
    logger.info("Creating coordinates DataFrame and adding corresponding zones.")
    df = pd.DataFrame(coords, columns=[lon_col, lat_col])
    
    gdf = add_zones_to_df(df, shapefile_path, crs, lon_col, lat_col, zone_col, new_zone_col)
    
    if gdf is None:
        return None
    
    
    logger.info("Successfully added zones to %s coordinates.", len(gdf))
    return [{lat_col:row[1][lat_col], lon_col:row[1][lon_col], new_zone_col:row[1][new_zone_col]} for row in gdf.iterrows()]


def load_preprocessed_data(fire_data_path: str, dispatch_data_path: str):
    """
    Load preprocessed fire and dispatch data into DataFrames.

    Parameters:
        fire_data_path (str): Path to the fire data CSV file.
        dispatch_data_path (str): Path to the dispatch data CSV file.

    Returns:
        pd.DataFrame: The fire DataFrame.
        pd.DataFrame: The dispatch DataFrame.
    """
    # --- Call args (debug) ---
    logger.debug("called with fire_data_path=%s, dispatch_data_path=%s", fire_data_path, dispatch_data_path)
    
    
    # --- Validate input ---
    try:
        if isinstance(fire_data_path, str) == False:
            raise TypeError("fire_data_path must be a string.")
        if isinstance(dispatch_data_path, str) == False:
            raise TypeError("dispatch_data_path must be a string.")
        if fire_data_path.endswith(".csv") == False:
            raise ValueError("fire_data_path must be a .csv file.")
        if dispatch_data_path.endswith(".csv") == False:
            raise ValueError("dispatch_data_path must be a .csv file.")
        if os.path.exists(fire_data_path) == False:
            raise FileNotFoundError(f"CSV file not found: {fire_data_path}")
        if os.path.exists(dispatch_data_path) == False:
            raise FileNotFoundError(f"CSV file not found: {dispatch_data_path}")
    except Exception:
        logger.exception("Failed while validating fire_data_path and dispatch_data_path.")
        raise
    
    
    # --- Load data ---
    logger.info("Loading fire and dispatch data from %s and %s.", fire_data_path, dispatch_data_path)
    try:
        fire_df = pd.read_csv(fire_data_path)
    except Exception:
        logger.exception("Failed while loading fire data.")
        raise
    try:
        dispatch_df = pd.read_csv(dispatch_data_path)
    except Exception:
        logger.exception("Failed while loading dispatch data.")
        raise
    
    
    # --- Convert datetime columns ---
    fire_datetime_cols = [col for col in fire_df.columns if 'datetime' in col]
    dispatch_datetime_cols = [col for col in dispatch_df.columns if col.startswith('hr_') or 'datetime' in col]
    
    if len(fire_datetime_cols) == 0:
        logger.warning("No datetime columns found in fire_df.")
    if len(dispatch_datetime_cols) == 0:
        logger.warning("No datetime columns found in dispatch_df.")
        
    logger.info("Converting %s columns in fire_df and %s columns in dispatch_df to datetime.", len(fire_datetime_cols), len(dispatch_datetime_cols))
    
    try:
        for col in fire_datetime_cols:
            fire_df[col] = pd.to_datetime(fire_df[col])
    except Exception:
        logger.exception("Failed to convert fire datetime columns.")
        raise
    try:
        for col in dispatch_datetime_cols:
            dispatch_df[col] = pd.to_datetime(dispatch_df[col])
    except Exception:
        logger.exception("Failed to convert dispatch datetime columns.")
        raise
        
    
    logger.info("Successfully converted %s columns in fire_df and %s columns in dispatch_df to datetime.", len(fire_datetime_cols), len(dispatch_datetime_cols))
    return fire_df, dispatch_df


def classify_response_type(
    dispatch_df: pd.DataFrame,
    mapping_template=(('BHA','H'), ('AA','A'), ('BA','T')),  # priority order
    response_column='glosa',
    new_response_column='recurso',
    remove_other=True,
    keep_old_column=True,
    case=True,
):
    """
    Create a new column in the dispatch dataframe based on a mapping template and classify the response type.
    
    Parameters:
        dispatch_df (pd.DataFrame): The DataFrame containing the dispatch data.
        mapping_template (tuple): The tuple of (pattern, label) pairs to classify the response type, where pattern is a regular expression pattern and label is the response type to assign.
        response_column (str): The name of the column containing the response type.
        new_response_column (str): The name of the column to add to the DataFrame to store the response type.
        remove_other (bool): Whether to remove the 'OTRO' response type.
        keep_old_column (bool): Whether to keep the original response type column.
        case (bool): Whether to match the response type case-sensitively.
    Returns:
        pd.DataFrame: The updated dispatch DataFrame.
    """
    
    #--- Call args (debug) ---
    logger.debug("called with dispatch_df.shape=%s, mapping_template=%s, response_column=%s, new_response_column=%s, remove_other=%s, keep_old_column=%s, case=%s", dispatch_df.shape, mapping_template, response_column, new_response_column, remove_other, keep_old_column, case)
    
    
    #--- Validate input ---
    try:
        if not isinstance(dispatch_df, pd.DataFrame):
            raise TypeError(f"dispatch_df must be a pandas DataFrame, got {type(dispatch_df).__name__}.")
        if not isinstance(mapping_template, tuple):
            raise TypeError(f"mapping_template must be a tuple, got {type(mapping_template).__name__}.")
        if not isinstance(response_column, str):
            raise TypeError(f"response_column must be a string, got {type(response_column).__name__}.")
        if not isinstance(new_response_column, str):
            raise TypeError(f"new_response_column must be a string, got {type(new_response_column).__name__}.")
        if not isinstance(remove_other, bool):
            raise TypeError(f"remove_other must be a boolean, got {type(remove_other).__name__}.")
        if not isinstance(keep_old_column, bool):
            raise TypeError(f"keep_old_column must be a boolean, got {type(keep_old_column).__name__}.")
        if not isinstance(case, bool):
            raise TypeError(f"case must be a boolean, got {type(case).__name__}.")
    except Exception:
        logger.exception("Invalid input to classify_response_type.")
        raise
    
    
    #--- Classify response type ---
    df = dispatch_df.copy()
    s = df[response_column].astype(str)

    # start as 'OTRO', then overwrite on first matching rule
    df[new_response_column] = 'OTRO'
    for pattern, label in mapping_template:
        mask = s.str.contains(pattern, case=case, na=False)
        # only assign where still OTRO to preserve priority
        df.loc[mask & (df[new_response_column] == 'OTRO'), new_response_column] = label
    
    if remove_other:
        df = df[df[new_response_column] != 'OTRO']
        logger.info("Successfully removed 'OTRO' response type.")

    if not keep_old_column:
        df = df.drop(columns=[response_column])
        logger.info("Successfully removed original response type column.")

    logger.info("Successfully classified response type, new column %s contains %s rows, old column %s contains %s rows.", new_response_column, df.shape[0], response_column, s.shape[0])
    return df

def print_missing_values(df):

    print("Number of missing values:")
    cols = [key for key in df.columns if "hr" in key]
    print(df[cols+["recurso"]].isnull().sum())

def add_first_response_arrival(
    fire_df: pd.DataFrame,
    dispatch_df: pd.DataFrame,
    fire_id_column: str = "fire_id",
    response_column: str = "recurso",
    response_types: list[str] = ["H", "T"],
    arrival_time_column: str = "hr_arribo",
    new_arrival_time_column: str = "arrival_datetime_des",
) -> pd.DataFrame:
    """
    Filter the dispatch dataframe to only include records with the first response of the specified types.

    Parameters:
        dispatch_df (pd.DataFrame): The DataFrame containing the dispatch data.
        response_column (str): The name of the column containing the response type.
        reponse_types (list[str]): The list of response types to filter by.
    Returns:
        pd.DataFrame: The filtered dispatch DataFrame.
    """
    # --- Call args (debug) ---
    logger.debug("called with dispatch_df.shape=%s", dispatch_df.shape)

    # --- Validate input ---
    try:
        if not isinstance(fire_df, pd.DataFrame):
            raise TypeError(f"fire_df must be a pandas DataFrame, got {type(fire_df).__name__}.")
        if not isinstance(dispatch_df, pd.DataFrame):
            raise TypeError(f"dispatch_df must be a pandas DataFrame, got {type(dispatch_df).__name__}.")
        
        if not isinstance(fire_id_column, str):
            raise TypeError(f"fire_id_column must be a string, got {type(fire_id_column).__name__}.")
        if fire_id_column not in fire_df.columns:
            raise ValueError(f"fire_id_column {fire_id_column} not in fire_df.columns")
        
        if not isinstance(response_column, str):
            raise TypeError(f"response_column must be a string, got {type(response_column).__name__}.")
        if response_column not in dispatch_df.columns:
            raise ValueError(f"response_column {response_column} not in dispatch_df.columns")
        
        if not isinstance(response_types, list):
            raise TypeError(f"response_types must be a list, got {type(response_types).__name__}.")
        if any(not isinstance(x, str) for x in response_types):
            raise TypeError(f"response_types must be a list of strings, got {type(response_types[0]).__name__}.")
        
        if not isinstance(arrival_time_column, str):
            raise TypeError(f"arrival_time_column must be a string, got {type(arrival_time_column).__name__}.")
        if arrival_time_column not in dispatch_df.columns:
            raise ValueError(f"arrival_time_column {arrival_time_column} not in dispatch_df.columns")
        
        if not isinstance(new_arrival_time_column, str):
            raise TypeError(f"new_arrival_time_column must be a string, got {type(new_arrival_time_column).__name__}.")
        if new_arrival_time_column in dispatch_df.columns:
            raise ValueError(f"new_arrival_time_column {new_arrival_time_column} already in dispatch_df.columns")
        
        if len(dispatch_df[dispatch_df["start_datetime"] > dispatch_df["hr_arribo"]]) != 0:
            raise ValueError("There are records with start_datetime > hr_arribo")
    except Exception:
        logger.exception("Failed to filter dispatchs by first response")
        raise

    # --- Filter by response type ---
    filtered_dispatch_df = dispatch_df[dispatch_df[response_column].isin(response_types)]
    first_response_df = (
        filtered_dispatch_df.groupby(fire_id_column)[arrival_time_column]
        .min()
        .reset_index()
        .rename(columns={arrival_time_column: new_arrival_time_column})
    )
    
    if new_arrival_time_column in fire_df.columns:
        fire_df = fire_df.drop(columns=[new_arrival_time_column])

    out_df = pd.merge(fire_df, first_response_df, on=fire_id_column, how='left')
    
    
    if "arrival_datetime_inc" in out_df.columns:
        s = (out_df["arrival_datetime_inc"] - out_df[new_arrival_time_column]).dropna().dt.total_seconds() / 60
        if (s > 0).any():
            raise ValueError("arrival_datetime_inc > arrival_datetime_des for some rows")
    if "start_datetime" in out_df.columns:
        if (out_df["start_datetime"] > out_df.get("arrival_datetime_inc")).dropna().any():
            raise ValueError("start_datetime > arrival_datetime_inc")
        if (out_df["start_datetime"] > out_df[new_arrival_time_column]).dropna().any():
            raise ValueError("start_datetime > arrival_datetime_des")

    logger.info("Added %s conatining the first response arrival time for selected response types.", new_arrival_time_column)
    return out_df

def check_for_outlier_fires(
    fire_df,
    outlier_tolerance=50000,
    start_column="start_datetime",
    fire_arrival_column="arrival_datetime_inc",
    response_arrival_column="arrival_datetime_des",
    fire_id_column="fire_id",
):
    """
    Check for outlier fires based on the difference between the start time of the fire and the arrival time of the first response.

    Parameters:
        fire_df (pd.DataFrame): The DataFrame containing the fire data.
        outlier_tolerance (int): The tolerance for identifying outlier fires (in minutes).
        start_column (str): The name of the column containing the start time from the dispatch dataframe.
        fire_arrival_column (str): The name of the column containing the arrival time from the fire dataframe.
        response_arrival_column (str): The name of the column containing the arrival time from the dispatch dataframe.
        fire_id_column (str): The name of the column containing the fire ID.
    Returns:
        list[str]: A list of outlier fire IDs.
    """
    # --- Call args (debug) ---
    logger.debug(
        "called with fire_df.shape=%s, outlier_tolerance=%s, start_column=%s, fire_arrival_column=%s, response_arrival_column=%s, fire_id_column=%s",
        fire_df.shape,
        outlier_tolerance,
        start_column,
        fire_arrival_column,
        response_arrival_column,
        fire_id_column,
    )


    # --- Validate input ---
    try:
        if not isinstance(fire_df, pd.DataFrame):
            raise TypeError(f"fire_df must be a pandas DataFrame, got {type(fire_df).__name__}.")
        
        if not isinstance(outlier_tolerance, int):
            raise TypeError(f"outlier_tolerance must be an integer, got {type(outlier_tolerance).__name__}.")
        if outlier_tolerance < 0:
            raise ValueError(f"outlier_tolerance must be a positive integer, got {outlier_tolerance}.")
        
        if not isinstance(start_column, str):
            raise TypeError(f"start_column must be a string, got {type(start_column).__name__}.")
        if start_column not in fire_df.columns:
            raise ValueError(f"start_column {start_column} not in fire_df.columns")
        
        if not isinstance(fire_arrival_column, str):
            raise TypeError(f"fire_arrival_column must be a string, got {type(fire_arrival_column).__name__}.")
        if fire_arrival_column not in fire_df.columns:
            raise ValueError(f"fire_arrival_column {fire_arrival_column} not in fire_df.columns")
        
        if not isinstance(response_arrival_column, str):
            raise TypeError(f"response_arrival_column must be a string, got {type(response_arrival_column).__name__}.")
        if response_arrival_column not in fire_df.columns:
            raise ValueError(f"response_arrival_column {response_arrival_column} not in fire_df.columns")
        
        if not isinstance(fire_id_column, str):
            raise TypeError(f"fire_id_column must be a string, got {type(fire_id_column).__name__}.")
        if fire_id_column not in fire_df.columns:
            raise ValueError(f"fire_id_column {fire_id_column} not in fire_df.columns")        
    except Exception as e:
        logger.exception("Invalid input to check_for_outlier_fires: %s", e)
        raise


    # --- Check for outliers ---
    s12 = ((fire_df[fire_arrival_column] - fire_df[response_arrival_column]).dt.total_seconds() / 60).dropna()
    s31 = ((fire_df[start_column] - fire_df[fire_arrival_column]).dt.total_seconds() / 60).dropna()
    s32 = ((fire_df[start_column] - fire_df[response_arrival_column]).dt.total_seconds() / 60).dropna()
    
    
    try:
        if np.any(s12 > 0):
            raise ValueError("There are records with arrival_datetime_inc > arrival_datetime_des")
        if np.any(s31 > 0):
            raise ValueError("There are records with start_datetime > arrival_datetime_inc")
        if np.any(s32 > 0):
            raise ValueError("There are records with start_datetime > arrival_datetime_des")
    except Exception:
        logger.exception("Failed to add first dispatch arrival")
        raise

    # Filter outliers based on start time difference from first response from fire dataframe
    outlier_fire_ids = fire_df.loc[abs(s32) > outlier_tolerance, "fire_id"].unique()

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
    #--- Call args (debug) ---
    logger.debug("called with fire_df.shape=%s, dispatch_df.shape=%s", fire_df.shape, dispatch_df.shape)
    
    
    #--- Validate input ---
    try:
        if not isinstance(fire_df, pd.DataFrame):
            raise TypeError(f"fire_df must be a pandas DataFrame, got {type(fire_df).__name__}.")
        if not isinstance(dispatch_df, pd.DataFrame):
            raise TypeError(f"dispatch_df must be a pandas DataFrame, got {type(dispatch_df).__name__}.")
    except Exception:
        logger.exception("fire_df and dispatch_df must be pandas DataFrames.")
        raise
    
    
    #--- Filter common IDs ---
    common_ids = set(fire_df['fire_id']).intersection(set(dispatch_df['fire_id']))
    
    if len(common_ids) == 0:
        logger.warning("There are no common fire IDs")
        
    fire_df = fire_df[fire_df['fire_id'].isin(common_ids)]
    dispatch_df = dispatch_df[dispatch_df['fire_id'].isin(common_ids)]
    
    
    logger.info("Successfully filtered common fire IDs.")
    logger.info("\tThere are %s fires with common IDs", len(fire_df))
    logger.info("\tThere are %s dispatches with common IDs", len(dispatch_df))
    return fire_df, dispatch_df
