import logging
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import richdem as rd
from pyproj import Transformer
from tqdm import tqdm
from aywen.utils import TimeOfDayFeatures


logger = logging.getLogger("aywen_logger")




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


#### Feature engineering 2_2 ####

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


#### Feature engineering 2_3 ####

def process_dt_minutes(df: pd.DataFrame, difference_column: str = 'dt_minutes', start_time_column: str = 'start_datetime', arrival_time_column: str = 'arrival_datetime_inc') -> pd.DataFrame:
    """
    Compute the time difference in minutes between two datetime columns
    and add both a numeric column and a categorical binned version.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    difference_column : str
        Name of the new numeric column with the time difference in minutes.
    start_time_column, arrival_time_column : str
        Names of the datetime columns; result = arrival_time_column - start_time_column.

    Returns
    -------
    pd.DataFrame
        DataFrame with two new columns:
        - difference_column: numeric difference in minutes
        - f"{difference_column}_cat": categorical bins (0-10, 10-30, 30+)
    """
    logger.debug("called with df.shape=%s, difference_column=%s, start_time_column=%s, arrival_time_column=%s", df.shape, difference_column, start_time_column, arrival_time_column)

    # validate
    if start_time_column not in df.columns or arrival_time_column not in df.columns:
        raise KeyError(f"Both {start_time_column} and {arrival_time_column} must exist in df.columns")

    out = df.copy()
    out[start_time_column] = pd.to_datetime(out[start_time_column])
    out[arrival_time_column] = pd.to_datetime(out[arrival_time_column])

    out[difference_column] = (out[arrival_time_column] - out[start_time_column]).dt.total_seconds() / 60.0

    # add categorical bins
    bins = [0, 10, 30, np.inf]
    labels = ['0-10', '10-30', '30+']
    out[f"{difference_column}_cat"] = pd.cut(out[difference_column], bins=bins, labels=labels, right=False)

    logger.info("Added %s and %s_cat columns.", difference_column, difference_column)
    logger.debug("Summary for %s:\n%s", difference_column, out[difference_column].describe())

    return out

def add_initial_derived_parameters(
    df: pd.DataFrame,
    surface_ha_col: str = "initial_surface_ha",
    fuel_col: str = "initial_fuel",
    m2_col: str = "initial_surface_m2",
    fuel_reduced_col: str = "initial_fuel_reduced",
    radius_col: str = "initial_radius_m",
    min_fuel_samples: int = 15,
    outlier_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Add initial derived parameters:
      1) Convert initial surface from hectares to m².
      2) Compute initial radius (m) assuming circular area.
      3) Trim upper outliers on surface area.
      4) Reduce/normalize fuel categories (rare -> 'Other', then slugify-ish).

    Parameters
    ----------
    df : pd.DataFrame
    surface_ha_col : str
        Column with initial area in hectares.
    fuel_col : str
        Column with initial fuel labels.
    m2_col : str
        Name of output area column in square meters.
    fuel_reduced_col : str
        Name of reduced/normalized fuel column.
    radius_col : str
        Name of output radius column (meters).
    min_fuel_samples : int
        Minimum count to keep a fuel as-is; rarer ones become 'Other'.
    outlier_quantile : float
        Upper quantile used to trim extreme m² outliers (default 0.99).

    Returns
    -------
    pd.DataFrame
        Copy of df with new columns and outliers removed.
    """
    if surface_ha_col not in df.columns:
        raise KeyError(f"Column '{surface_ha_col}' not found in df.")
    if fuel_col not in df.columns:
        raise KeyError(f"Column '{fuel_col}' not found in df.")

    out = df.copy()

    # 1) hectares -> m²
    out[m2_col] = pd.to_numeric(out[surface_ha_col], errors="coerce") * 10_000
    logger.info("Created %s from %s (ha→m²).", m2_col, surface_ha_col)
    logger.debug("Summary %s:\n%s", m2_col, out[m2_col].describe())

    # 2) Initial radius (assuming circular area)
    out[radius_col] = np.sqrt(out[m2_col] / np.pi)
    logger.info("Computed %s from %s.", radius_col, m2_col)

    # 3) Trim upper outliers
    upper_limit = out[m2_col].quantile(outlier_quantile)
    before = len(out)
    out = out[out[m2_col] < upper_limit].reset_index(drop=True)
    logger.info(
        "Trimmed upper outliers at q=%.3f (%.2f). Rows: %d → %d.",
        outlier_quantile, upper_limit, before, len(out)
    )

    # 4) Reduce/normalize fuel categories
    fuel_counts = out[fuel_col].value_counts(dropna=True)
    rare_fuels = fuel_counts[fuel_counts < min_fuel_samples].index
    out[fuel_reduced_col] = out[fuel_col].replace(rare_fuels, "Other")

    out[fuel_reduced_col] = out[fuel_reduced_col].map(
        lambda x: str(x)
            .replace("-", "")
            .replace(" ", "_")
            .replace(",", "_")
            .replace(">", "gt_")
            .replace("<", "lt_")
            .replace("=", "eq_")
            .replace(".", "")
            .lower()
    )

    kept = out[fuel_reduced_col].nunique(dropna=True)
    original = df[fuel_col].nunique(dropna=True)
    logger.info("Fuel categories reduced: %d (from %d).", kept, original)

    return out


def get_weather_data(
    folder: str = "G:/Shared drives/OpturionHome/AraucoFire/2_data/processed/",
    filename: str = "Incendios_2014-2025_VHT_v3.csv",
    usecols: list[str] | None = None,
    sep: str = ";",
) -> pd.DataFrame:
    """
    Load and clean weather data for fires.

    - Reads selected columns from the CSV
    - Renames columns to snake_case
    - Converts numeric columns with comma decimals to floats
    - (Optional) Visualizes missingness with missingno.matrix

    Parameters
    ----------
    folder : str
        Directory containing the CSV file.
    filename : str
        CSV file name.
    usecols : list[str] | None
        Columns to read from the CSV. If None, uses the standard set.
    sep : str
        CSV delimiter.

    Returns
    -------
    pd.DataFrame
        Cleaned weather DataFrame.
    """
    if usecols is None:
        usecols = [
            "FICHA",
            "TEMPERATURA",
            "HUMEDAD_RELATIVA",
            "VELOCIDAD_VIENTO_MS",
            "VELOCIDAD_VIENTO_KNOTS",
            "DIRVIENTO",
        ]

    path = os.path.join(folder, filename)
    logger.debug("get_weather_data called with path=%s, usecols=%s, sep=%s", path, usecols, sep)

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    # Load
    weather = pd.read_csv(path, sep=sep, usecols=usecols)

    # Rename
    weather = weather.rename(
        columns={
            "FICHA": "fire_id",
            "VELOCIDAD_VIENTO_MS": "wind_speed_ms",
            "VELOCIDAD_VIENTO_KNOTS": "wind_speed_knots",
            "DIRVIENTO": "wind_direction",
            "HUMEDAD_RELATIVA": "relative_humidity",
            "TEMPERATURA": "temperature",
        }
    )

    # Numeric cleanup: comma decimals -> dot, then to float
    num_cols = ["wind_speed_ms", "wind_speed_knots", "wind_direction", "relative_humidity", "temperature"]
    for c in num_cols:
        if c in weather.columns:
            weather[c] = pd.to_numeric(
                weather[c].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    logger.info("Loaded weather data: %s rows, %s cols", weather.shape[0], weather.shape[1])
    logger.debug("Weather numeric summary:\n%s", weather[num_cols].describe(include="all"))

    return weather

def merge_weather_data(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fire_id_col: str = "fire_id",
) -> pd.DataFrame:
    """
    Merge weather data into the fire dataframe, dropping any pre-existing
    weather columns in df to avoid collisions. No transformations here.
    """
    if fire_id_col not in df.columns:
        raise KeyError(f"'{fire_id_col}' not in df.columns")
    if fire_id_col not in weather_df.columns:
        raise KeyError(f"'{fire_id_col}' not in weather_df.columns")

    out = df.copy()

    # Drop any weather/derived cols so re-running stays idempotent
    cols_to_drop = [
        "temperature", "relative_humidity", "wind_direction",
        "wind_speed_ms", "wind_speed_knots",
        "wind_speed_mm", "wind_speed_kmh",
        "wind_speed_cat", "temperature_cat", "humidity_cat",
        "weather_index", "weather_index_full",
    ]
    existing = [c for c in cols_to_drop if c in out.columns]
    if existing:
        out.drop(columns=existing, inplace=True)
        logger.info("Dropped pre-existing weather columns: %s", existing)

    # Merge
    out = out.merge(weather_df, on=fire_id_col, how="left")
    logger.info("Merged weather data: result shape %s", out.shape)
    return out

def process_weather_features(
    df: pd.DataFrame,
    *,
    humidity_bounds: tuple[float, float] = (0.0, 100.0),
    wind_dir_bounds: tuple[float, float] = (0.0, 360.0),
    speed_kmh_threshold: float = 30.0,
    temp_c_threshold: float = 30.0,
    humidity_threshold: float = 30.0,
    drop_raw_wind: bool = True,
) -> pd.DataFrame:
    """
    Clean and engineer weather features on a dataframe that already contains
    the weather columns. Performs:
      - Clamp invalid relative_humidity and wind_direction to NaN.
      - Fill wind_speed_ms from knots (1 kt = 0.51445 m/s) if missing.
      - Derive wind_speed_mm (m/min) and wind_speed_kmh (km/h).
      - Build 30–30–30 categories and combined indices.

    Expects columns (some may be missing): temperature, relative_humidity,
    wind_direction, wind_speed_ms, wind_speed_knots.
    """
    out = df.copy()

    # Ensure numeric types if present
    for c in ["temperature", "relative_humidity", "wind_direction", "wind_speed_ms", "wind_speed_knots"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Clamp invalid ranges
    if "relative_humidity" in out.columns:
        lo, hi = humidity_bounds
        out["relative_humidity"] = out["relative_humidity"].where(
            (out["relative_humidity"] >= lo) & (out["relative_humidity"] <= hi), np.nan
        )

    if "wind_direction" in out.columns:
        lo, hi = wind_dir_bounds
        out.loc[(out["wind_direction"] < lo) | (out["wind_direction"] > hi), "wind_direction"] = np.nan

    # Fill wind_speed_ms from knots if needed; then derive speeds
    has_ms = "wind_speed_ms" in out.columns
    has_knots = "wind_speed_knots" in out.columns

    if has_ms or has_knots:
        if not has_ms:
            out["wind_speed_ms"] = np.nan
            has_ms = True

        if has_knots:
            out["wind_speed_knots"] = pd.to_numeric(out["wind_speed_knots"], errors="coerce")
            out["wind_speed_ms"] = out["wind_speed_ms"].fillna(0.51445 * out["wind_speed_knots"])

        # Derived speeds (only if ms exists)
        out["wind_speed_mm"]  = out["wind_speed_ms"] * 60.0   # m/min
        out["wind_speed_kmh"] = out["wind_speed_ms"] * 3.6    # km/h

        if drop_raw_wind:
            for c in ("wind_speed_knots", "wind_speed_ms"):
                if c in out.columns:
                    out.drop(columns=c, inplace=True)

    # 30–30–30 categories
    if "wind_speed_kmh" in out.columns:
        out["wind_speed_cat"] = pd.cut(
            out["wind_speed_kmh"],
            bins=[0, speed_kmh_threshold, np.inf],
            labels=["S", "F"],
            right=False,
        )
    else:
        out["wind_speed_cat"] = pd.Categorical([np.nan] * len(out))

    if "temperature" in out.columns:
        out["temperature_cat"] = pd.cut(
            out["temperature"],
            bins=[0, temp_c_threshold, np.inf],
            labels=["L", "H"],
            right=False,
        )
    else:
        out["temperature_cat"] = pd.Categorical([np.nan] * len(out))

    if "relative_humidity" in out.columns:
        out["humidity_cat"] = pd.cut(
            out["relative_humidity"],
            bins=[0, humidity_threshold, np.inf],
            labels=["D", "W"],
            right=False,
        )
    else:
        out["humidity_cat"] = pd.Categorical([np.nan] * len(out))

    # Fill missing cats with 'N'
    for c in ["wind_speed_cat", "temperature_cat", "humidity_cat"]:
        out[c] = out[c].cat.add_categories("N").fillna("N")

    # Combined indices
    out["weather_index"] = np.where(
        (out["wind_speed_cat"] != "N") & (out["temperature_cat"] != "N"),
        out["wind_speed_cat"].astype(str) + out["temperature_cat"].astype(str),
        "NA",
    )

    out["weather_index_full"] = np.where(
        (out["wind_speed_cat"] != "N")
        & (out["temperature_cat"] != "N")
        & (out["humidity_cat"] != "N"),
        out["wind_speed_cat"].astype(str)
        + out["temperature_cat"].astype(str)
        + out["humidity_cat"].astype(str),
        "NA",
    )

    logger.info("process_weather_features completed.")
    return out


def compute_ffdi(
    df: pd.DataFrame,
    *,
    temp_col: str = "temperature",        # °C
    wind_kmh_col: str = "wind_speed_kmh", # km/h @10 m
    rh_col: str = "relative_humidity",    # %
    drought: float | pd.Series = 10.0,    # scalar in [0,10] or Series aligned to df.index
    out_col: str = "ffdi",
    attach: bool = True,
    clip_min: float | None = 0.0,
) -> pd.Series | pd.DataFrame:
    """
    Compute McArthur Mk5 Forest Fire Danger Index (FFDI).

    FFDI = 2 * exp( -0.45 + 0.987*ln(D) - 0.0345*RH + 0.0338*T + 0.0234*V )

    where:
      D  = drought factor (0–10)
      RH = relative humidity (%)
      T  = air temperature (°C)
      V  = 10 m wind speed (km/h)

    If `attach=True`, returns a DataFrame with `out_col` added; otherwise returns the Series.
    """
    logger.debug("compute_ffdi called with temp_col=%s, wind_kmh_col=%s, rh_col=%s, out_col=%s, attach=%s",
                 temp_col, wind_kmh_col, rh_col, out_col, attach)

    # validate columns
    for c in (temp_col, wind_kmh_col, rh_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in df.")

    T = pd.to_numeric(df[temp_col], errors="coerce").to_numpy()
    V = pd.to_numeric(df[wind_kmh_col], errors="coerce").to_numpy()
    RH = pd.to_numeric(df[rh_col], errors="coerce").to_numpy()

    # drought factor: scalar or Series
    if np.isscalar(drought):
        D = np.full_like(T, float(drought), dtype=float)
    else:
        D = pd.to_numeric(pd.Series(drought).reindex(df.index), errors="coerce").to_numpy()

    # protect ln(D)
    eps = 1e-6
    D_safe = np.maximum(D, eps)

    ffdi_vals = 2.0 * np.exp(-0.45 + 0.987 * np.log(D_safe) - 0.0345 * RH + 0.0338 * T + 0.0234 * V)

    if clip_min is not None:
        ffdi_vals = np.maximum(ffdi_vals, clip_min)

    s = pd.Series(ffdi_vals, index=df.index, name=out_col)
    logger.info("Computed FFDI: non-null=%d, min=%.3f, p50=%.3f, max=%.3f",
                s.notna().sum(), float(np.nanmin(s)), float(np.nanmedian(s)), float(np.nanmax(s)))

    if attach:
        out = df.copy()
        out[out_col] = s
        return out
    return s

def categorize_ffdi(
    ffdi: pd.Series,
    *,
    out_col: str = "ffdi_category"
) -> pd.Series:
    """
    Australian FFDI categories (forest):
      0–11  Low–Moderate
      12–24 High
      25–49 Very High
      50–99 Severe
      100–149 Extreme
      150+  Catastrophic
    """
    bins = [-np.inf, 11, 24, 49, 99, 149, np.inf]
    labels = ["Low–Moderate", "High", "Very High", "Severe", "Extreme", "Catastrophic"]
    cat = pd.cut(ffdi, bins=bins, labels=labels, right=True)
    cat.name = out_col
    logger.info("Categorized FFDI into %d levels.", len(labels))
    return cat



def propagation_speed_analysis(
    df: pd.DataFrame,
    radius_col: str = "initial_radius_m",
    dt_col: str = "dt_minutes",
    speed_col: str = "propagation_speed_mm",   # meters per minute
    add_kmh: bool = True,
    speed_kmh_col: str = "propagation_speed_kmh",
    lower_limit: float = 0.1,                  # m/min
    upper_limit: float = 100.0,                # m/min
    return_summary: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """
    Compute semi-major-axis propagation speed and log diagnostics.

    speed (m/min) = initial_radius_m / dt_minutes

    - Logs counts of negative/zero/NaN/∞ speeds
    - Logs counts below `lower_limit` and above `upper_limit`
    - Optionally adds a km/h column: km/h = (m/min) * 0.06
    - Does not subset rows; leaves filtering to downstream steps
    """
    # validate
    if radius_col not in df.columns:
        raise KeyError(f"Column '{radius_col}' not found in df.")
    if dt_col not in df.columns:
        raise KeyError(f"Column '{dt_col}' not found in df.")

    out = df.copy()
    out[radius_col] = pd.to_numeric(out[radius_col], errors="coerce")
    out[dt_col] = pd.to_numeric(out[dt_col], errors="coerce")

    # compute speed (m/min)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[speed_col] = out[radius_col] / out[dt_col]

    # diagnostics
    n_neg = int((out[speed_col] < 0).sum())
    n_zero = int((out[speed_col] == 0).sum())
    n_nan = int(out[speed_col].isna().sum())
    n_inf = int(np.isinf(out[speed_col]).sum())
    logger.info(
        "Propagation speed diagnostics [%s]: neg=%d, zero=%d, nan=%d, inf=%d",
        speed_col, n_neg, n_zero, n_nan, n_inf
    )

    # summary stats
    summary = out[speed_col].replace([np.inf, -np.inf], np.nan).describe()
    logger.debug("Summary for %s:\n%s", speed_col, summary)

    # threshold counts (no filtering here)
    below = int((out[speed_col] < lower_limit).sum())
    above = int((out[speed_col] > upper_limit).sum())
    logger.info(
        "Threshold counts for %s: < %.3f => %d rows, > %.3f => %d rows",
        speed_col, lower_limit, below, upper_limit, above
    )

    # optional km/h
    if add_kmh:
        # m/min → km/h: multiply by 0.06
        out[speed_kmh_col] = out[speed_col] * 0.06

    return (out, summary) if return_summary else out



def dem_to_df(
    dem_path: str,
    df: pd.DataFrame,
    *,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    id_col: str = "fire_id",
    points_crs: str = "EPSG:4326",
    slope_units: str = "degrees",   # "degrees" or "radians"
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Sample elevation, slope, and aspect from a DEM at point locations and return a tidy table.

    Returns a DataFrame with columns:
      [id_col, 'elevation', 'slope_degrees'|'slope_radians', 'aspect_degrees'].
    """
    logger.info("Loading DEM from %s", dem_path)
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    dem = rd.LoadGDAL(dem_path)  # rdarray with .projection, .geotransform, .shape
    nrows, ncols = dem.shape
    gt = dem.geotransform

    slope_attr = "slope_degrees" if slope_units.lower().startswith("deg") else "slope_radians"
    logger.info("Computing terrain attributes (%s, aspect)...", slope_attr)
    slope = rd.TerrainAttribute(dem, attrib=slope_attr)
    aspect = rd.TerrainAttribute(dem, attrib="aspect")  # degrees [0, 360)

    transformer = Transformer.from_crs(points_crs, dem.projection, always_xy=True)

    ids, elev_vals, slope_vals, aspect_vals = [], [], [], []
    iterator = zip(df[id_col].to_numpy(), df[lon_col].to_numpy(), df[lat_col].to_numpy())
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Sampling DEM")

    for pt_id, lon, lat in iterator:
        try:
            x, y = transformer.transform(float(lon), float(lat))
            # col = (x - x0)/px_w ; row = (y - y0)/px_h  (px_h typically negative)
            col = int(round((x - gt[0]) / gt[1]))
            row = int(round((y - gt[3]) / gt[5]))

            if 0 <= row < nrows and 0 <= col < ncols:
                elev = float(dem[row, col])
                slp  = float(slope[row, col])
                asp  = float(aspect[row, col])
            else:
                elev = slp = asp = np.nan
        except Exception as e:
            logger.debug("DEM sample error at id=%s lon=%s lat=%s: %s", pt_id, lon, lat, e)
            elev = slp = asp = np.nan

        ids.append(pt_id)
        elev_vals.append(elev)
        slope_vals.append(slp)
        aspect_vals.append(asp)

    slope_col = "slope_degrees" if slope_attr == "slope_degrees" else "slope_radians"
    topo_df = pd.DataFrame(
        {
            id_col: ids,
            "elevation": elev_vals,
            slope_col: slope_vals,
            "aspect_degrees": aspect_vals,
        }
    )
    logger.info(
        "Topography sampled for %d points (DEM %dx%d). Non-null elevation: %d",
        len(topo_df), nrows, ncols, topo_df["elevation"].notna().sum()
    )
    return topo_df
    
def get_topographical_data(
    df: pd.DataFrame,
    *,
    from_csv: bool = True,
    from_dem: bool = False,
    csv_path: str | None = None,
    dem_path: str | None = None,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    id_col: str = "fire_id",
    points_crs: str = "EPSG:4326",
    slope_units: str = "degrees",
    show_progress: bool = True,
    save_topo_csv: str | None = None,
) -> pd.DataFrame:
    """
    Obtain topographic data (elevation, slope, aspect) either by reading a precomputed CSV
    or by sampling a DEM, then merge into `df` on `id_col`.

    - If `from_dem=True`, requires `dem_path`; optionally saves the sampled table to `save_topo_csv`.
    - If `from_csv=True`, requires `csv_path`.

    Returns a new DataFrame with topography merged. Pre-existing topo columns
    ('elevation', 'slope_*', 'aspect_degrees') are dropped before merge to keep it idempotent.
    """
    if not (from_csv ^ from_dem):
        raise AssertionError("Exactly one of from_csv or from_dem must be True.")

    if from_dem:
        if not dem_path:
            raise ValueError("dem_path must be provided when from_dem=True.")
        topo_df = dem_to_df(
            dem_path,
            df,
            lon_col=lon_col,
            lat_col=lat_col,
            id_col=id_col,
            points_crs=points_crs,
            slope_units=slope_units,
            show_progress=show_progress,
        )
        if save_topo_csv:
            topo_df.to_csv(save_topo_csv, index=False)
            logger.info("Saved topographic table to %s", save_topo_csv)
    else:
        if not csv_path:
            raise ValueError("csv_path must be provided when from_csv=True.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Topographic CSV not found: {csv_path}")
        topo_df = pd.read_csv(csv_path)
        if id_col not in topo_df.columns:
            raise KeyError(f"Column '{id_col}' not found in topographic CSV.")

    # Determine slope column (degrees or radians)
    slope_cols = [c for c in topo_df.columns if c.startswith("slope_")]
    if len(slope_cols) == 0:
        raise KeyError("No slope column found in topographic data (expected 'slope_degrees' or 'slope_radians').")
    slope_col = slope_cols[0]

    # Drop existing topo cols in df
    cols_to_drop = ["elevation", slope_col, "aspect_degrees"]
    existing = [c for c in cols_to_drop if c in df.columns]
    out = df.copy()
    if existing:
        out.drop(columns=existing, inplace=True)
        logger.info("Dropped pre-existing topography columns: %s", existing)

    # Merge
    keep_cols = [id_col, "elevation", slope_col, "aspect_degrees"]
    missing = [c for c in keep_cols if c not in topo_df.columns]
    if missing:
        raise KeyError(f"Topographic data missing columns: {missing}")

    out = out.merge(topo_df[keep_cols], on=id_col, how="left")
    logger.info("Merged topographic data: result shape %s", out.shape)
    return out

    
    
# --- Angular helpers ---------------------------------------------------------
def _normalize_deg(a):
    """
    Normalize angles (deg) to [0, 360). Accepts pd.Series or array-like.
    Preserves Series index/name when provided.
    """
    if isinstance(a, pd.Series):
        vals = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
        vals = np.mod(vals, 360.0)
        vals[vals < 0] += 360.0
        return pd.Series(vals, index=a.index, name=a.name)
    vals = np.asarray(a, dtype=float)
    vals = np.mod(vals, 360.0)
    vals[vals < 0] += 360.0
    return vals

def aspect_to_upslope(aspect_deg: pd.Series) -> pd.Series:
    """
    RichDEM aspect is DOWNSLOPE azimuth in degrees [0,360); -1 for flats.
    Return UPSLOPE azimuth; flats -> NaN.
    """
    logger.debug("aspect_to_upslope called with %d values", len(aspect_deg))
    a = pd.to_numeric(aspect_deg, errors="coerce")
    a = a.where(a >= 0, np.nan)  # -1 (flat) -> NaN
    upslope = _normalize_deg(a + 180.0)
    upslope.name = "upslope_dir_deg"
    return upslope

def wind_from_to_dir(wind_from_deg: pd.Series) -> pd.Series:
    """
    Convert meteorological wind 'FROM' direction (deg, true) to 'TO' azimuth.
    """
    logger.debug("wind_from_to_dir called with %d values", len(wind_from_deg))
    w = _normalize_deg(pd.to_numeric(wind_from_deg, errors="coerce"))
    to_dir = _normalize_deg(w + 180.0)
    to_dir.name = "wind_to_dir_deg"
    return to_dir

def circular_diff_deg(a_deg: pd.Series, b_deg: pd.Series) -> pd.Series:
    """
    Smallest signed angular difference (a - b) in degrees ∈ [-180, 180].
    """
    logger.debug("circular_diff_deg called with %d values", len(a_deg))
    a = pd.to_numeric(a_deg, errors="coerce")
    b = pd.to_numeric(b_deg, errors="coerce")
    d = (a - b + 180.0) % 360.0 - 180.0
    d.name = None
    return d


def add_wind_slope_alignment(
    df: pd.DataFrame,
    wind_dir_col: str = "wind_direction",   # FROM direction (deg, true)
    wind_speed_col: str = "wind_speed_kmh", # km/h
    aspect_col: str = "aspect_degrees",     # downslope azimuth (deg)
    slope_col: str = "slope_degrees",       # slope angle (deg)
) -> pd.DataFrame:
    """
    Add wind–slope alignment features:

      upslope_dir_deg         : azimuth opposite to aspect (deg)
      wind_to_dir_deg         : wind 'to' direction (deg)
      wind_upslope_angle_deg  : absolute angle between wind-to and upslope [0..180]
      wind_upslope_cos        : cos(angle)  (+1 aligned, 0 cross, -1 opposed)
      wind_parallel_kmh       : V * cos(angle)   (>0 aids upslope; <0 opposes)
      wind_cross_kmh          : |V * sin(angle)| (km/h)
      slope_tan               : tan(slope_rad)
      slope_tan2              : tan(slope_rad)^2
      slope_wind_synergy      : slope_tan * max(cos(angle), 0)

    Returns a NEW DataFrame.
    """
    logger.debug(
        "add_wind_slope_alignment called with df.shape=%s, cols=[%s,%s,%s,%s]",
        df.shape, wind_dir_col, wind_speed_col, aspect_col, slope_col
    )

    # Validate presence of required columns
    for c in (wind_dir_col, wind_speed_col, aspect_col, slope_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in df.")

    out = df.copy()

    # Convert inputs to numeric (NaN-safe)
    out[wind_dir_col] = pd.to_numeric(out[wind_dir_col], errors="coerce")
    out[wind_speed_col] = pd.to_numeric(out[wind_speed_col], errors="coerce")
    out[aspect_col] = pd.to_numeric(out[aspect_col], errors="coerce")
    out[slope_col] = pd.to_numeric(out[slope_col], errors="coerce")

    # 1) Azimuths we need
    upslope = aspect_to_upslope(out[aspect_col])          # upslope azimuth (deg)
    wind_to = wind_from_to_dir(out[wind_dir_col])         # wind-to azimuth (deg)

    # 2) Angle & cosine between wind-to and upslope
    delta = circular_diff_deg(wind_to, upslope)           # signed in [-180, 180]
    delta_abs = delta.abs()
    delta_rad = np.deg2rad(delta_abs)                     # use absolute angle for components magnitude

    out["upslope_dir_deg"] = upslope
    out["wind_to_dir_deg"] = wind_to
    out["wind_upslope_angle_deg"] = delta_abs             # 0–180
    out["wind_upslope_cos"] = np.cos(np.deg2rad(delta))   # signed cosine: +1 aligned, -1 opposed

    # 3) Resolve wind into parallel / cross components
    V = out[wind_speed_col]
    out["wind_parallel_kmh"] = V * out["wind_upslope_cos"]      # >0 aids upslope; <0 opposes
    out["wind_cross_kmh"] = (V * np.sin(delta_rad)).abs()       # magnitude (km/h)

    # 4) Slope transforms & synergy
    slope_rad = np.deg2rad(out[slope_col])
    slope_tan = np.tan(slope_rad)
    out["slope_tan"] = slope_tan
    out["slope_tan2"] = slope_tan ** 2
    out["slope_wind_synergy"] = slope_tan * np.clip(out["wind_upslope_cos"], 0, None)

    # Diagnostics
    nn = int(out["wind_upslope_angle_deg"].notna().sum())
    logger.info("Wind–slope alignment added. Non-null angle rows: %d / %d", nn, len(out))

    return out
    
    


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

    len_df = len(df)
    len_gdf = len(gdf)
    if len_df != len_gdf:
        logger.info("Row count changed after spatial join: %d -> %d", len_df, len_gdf)

    len_gdf = len(gdf)
    gdf = gdf.dropna(subset=[zone_col])
    if len_gdf != len(gdf):
        logger.info("Dropped %d rows with no zone match.", len_gdf - len(gdf))
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
    separation_levels: list[str] = [['Costa', 'Valle', 'Cordillera'], ['1', '2', '3', '4', '5']]
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
            df[key] = pd.Categorical(
                split_cols[i], categories=separation_levels[i], ordered=True
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
 
    
    
def filter_communes_by_min_samples(
    df: pd.DataFrame,
    commune_col: str = "commune",
    min_samples: int = 10,
    coerce_str: bool = True,
) -> pd.DataFrame:
    """
    Keep only communes with at least `min_samples` observations.

    - Logs before/after counts.
    - Optionally coerces the commune column to string (matches your snippet).

    Parameters
    ----------
    df : pd.DataFrame
    commune_col : str
        Column holding commune names/ids.
    min_samples : int
        Minimum number of rows required to keep a commune.
    coerce_str : bool
        If True, cast the commune column to str after filtering.

    Returns
    -------
    pd.DataFrame
        Filtered copy of df.
    """
    if commune_col not in df.columns:
        raise KeyError(f"Column '{commune_col}' not found in df.")

    n_before = len(df)
    n_communes_before = df[commune_col].nunique(dropna=True)
    logger.info("Observed %,d observations across %d communes.", n_before, n_communes_before)

    counts = df[commune_col].value_counts()
    well_sampled = counts[counts >= min_samples].index

    out = df[df[commune_col].isin(well_sampled)].copy()
    if coerce_str:
        out[commune_col] = out[commune_col].astype(str)

    n_after = len(out)
    n_communes_after = out[commune_col].nunique(dropna=True)
    logger.info("Retained %,d observations across %d communes (min_samples=%d).",
                n_after, n_communes_after, min_samples)

    return out
    
    
def add_sin_cos_hour(
    df: pd.DataFrame,
    datetime_col: str = "start_datetime",
    hour_col: str = "hour",
    sin_col: str = "sin_hour",
    cos_col: str = "cos_hour",
    period: int = 24,
    tod: TimeOfDayFeatures | None = None,
) -> pd.DataFrame:
    """
    Add cyclical hour-of-day features (sin/cos) using TimeOfDayFeatures.

    Steps
    -----
    1) Extract hour-of-day from `datetime_col` into `hour_col` (0..23).
    2) Use TimeOfDayFeatures(period) to compute sin/cos encodings.
       - `sin_col` := sin(theta + phase)
       - `cos_col` := cos(theta + phase)
       where theta = 2π * hour / period

    Parameters
    ----------
    df : pd.DataFrame
    datetime_col : str
        Datetime column to derive hour from (coerced with pd.to_datetime).
    hour_col : str
        Output column to store extracted hour.
    sin_col, cos_col : str
        Output columns for the cyclical encoding.
    period : int
        Period of the cycle (default 24 for hours).
    tod : TimeOfDayFeatures | None
        If provided, use this instance; otherwise create one with (period).

    Returns
    -------
    pd.DataFrame
        Copy of df with `hour_col`, `sin_col`, and `cos_col` added.
    """
    logger.debug(
        "add_sin_cos_hour called with df.shape=%s, datetime_col=%s, period=%s",
        df.shape, datetime_col, period
    )

    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in df.")

    out = df.copy()

    # 1) Extract hour-of-day
    dt = pd.to_datetime(out[datetime_col], errors="coerce")
    out[hour_col] = dt.dt.hour

    # 2) Build/features via TimeOfDayFeatures
    if tod is None:
        tod = TimeOfDayFeatures(period=period)

    # The transformer accepts any shape; we’ll pass 2D to mirror your snippet
    hours_arr = out[hour_col].to_numpy().reshape(-1, 1)
    sin_time, cos_time = tod.transform(hours_arr)

    out[sin_col] = sin_time
    out[cos_col] = cos_time

    nn = int(out[hour_col].notna().sum())
    logger.info(
        "Added cyclical hour features (%s,%s) with period=%d. Non-null hours=%d/%d.",
        sin_col, cos_col, period, nn, len(out)
    )
    logger.debug("Hour summary:\n%s", out[hour_col].describe())

    return out

    
def add_subseason(
    df: pd.DataFrame,
    datetime_col: str = "start_datetime",
    month_col: str = "month",
    subseason_col: str = "subseason",
    categories: list[str] = ["off", "low", "medium", "high"],
    ordered: bool = True,
) -> pd.DataFrame:
    """
    Add a 'subseason' categorical based on month from `datetime_col`.

    Default mapping:
      - high   : Jan (1), Feb (2)
      - medium : Nov (11), Dec (12), Mar (3)
      - low    : Apr (4), Oct (10)
      - off    : all other months (May–Sep)

    Returns a NEW DataFrame with `month_col` and `subseason_col`.
    """
    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in df.")

    out = df.copy()
    dt = pd.to_datetime(out[datetime_col], errors="coerce")
    out[month_col] = dt.dt.month

    # month -> label (others/NaN -> 'off')
    month_to_label = {
        1: "high", 2: "high",
        3: "medium", 11: "medium", 12: "medium",
        4: "low", 10: "low",
    }
    out[subseason_col] = out[month_col].map(month_to_label).fillna("off")
    out[subseason_col] = pd.Categorical(out[subseason_col], categories=categories, ordered=ordered)

    logger.info(
        "Added subseason from %s. Counts: %s",
        datetime_col, out[subseason_col].value_counts(dropna=False).to_dict()
    )
    return out


def is_high_season(ts):
    """
    Returns high or low depending if the timestamp falls between Dec 15 and Mar 15 (inclusive),
    ignoring the year.
    """
    m, d = ts.month, ts.day
    
    # Dec 15 to Dec 31
    if (m == 12 and d >= 15) or (m == 1) or (m == 2) or (m == 3 and d <= 15):
        return 'high'
    else:
        return 'low'
    

def add_high_seasson(
        df: pd.DataFrame,
        datetime_col: str = "start_datetime",
        high_season_col: str = "high_season",
    ) -> pd.DataFrame:
    """
    Add a 'high_season' categorical based on month and day from `datetime_col`.

    Default mapping:
        - high   : Dec 15 to Mar 15 (inclusive)
        - low    : all other dates

    Returns a NEW DataFrame with `high_season` column.
    """
    datetime_col = "start_datetime"
    high_season_col = "high_season"

    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in df.")

    out = df.copy()
    dt = pd.to_datetime(out[datetime_col])
    out[high_season_col] = dt.apply(is_high_season)
    out[high_season_col] = pd.Categorical(out[high_season_col], categories=['low', 'high'], ordered=True)

    logger.info(
        "Added high_season from %s. Counts: %s",
        datetime_col, out[high_season_col].value_counts(dropna=False).to_dict()
    )
    return out


def nocturnal(row, target_zone_col: str = "target_zone") -> str:
  if row[target_zone_col]=="NOCTURNO":
    return "Night"
  else:
    return "Day"


def add_nocturnal(
        df: pd.DataFrame,
        target_zone_col: str = "target_zone",
        nocturnal_col: str = "nocturnal",
        ) -> pd.DataFrame:
    """
    Add a 'nocturnal' categorical based on the 'target_zone' column.

    Returns a NEW DataFrame with the 'nocturnal' column.
    """
    if target_zone_col not in df.columns:
        raise KeyError(f"Column '{target_zone_col}' not found in df.")

    out = df.copy()
    out[nocturnal_col] = out.apply(lambda row: nocturnal(row, target_zone_col=target_zone_col), axis=1)
    out[nocturnal_col] = pd.Categorical(out[nocturnal_col], categories=["Day", "Night"], ordered=True)

    logger.info(
        f"Added nocturnal from {target_zone_col}. Counts: %s",
        out[nocturnal_col].value_counts(dropna=False).to_dict()
    )
    return out

def feature_engineering_pipeline(
        df: pd.DataFrame,
        shapefile_path: str,
        csv_path: str,
        id_col: str = "fire_id",
        datetime_col: str = "start_datetime",
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        crs: str = "EPSG:4326",
        zone_col: str = "area",
        target_zone_col: str = "target_zone"
        ) -> pd.DataFrame:

        out = df.copy()
        out = add_zones_to_df(df = out,shapefile_path=shapefile_path,crs = crs,lon_col=lon_col,lat_col=lat_col,zone_col=zone_col,new_zone_col="zone_alert",)
        out = split_zones(df = out,zone_col = "zone_alert",separation_keys = ["zone_WE", "zone_NS"]) 
        out = add_sin_cos_hour(df = out,datetime_col = datetime_col,sin_col = 'sin_hour',cos_col = 'cos_hour')
        out = get_topographical_data(df = out,csv_path = csv_path,lon_col = lon_col,lat_col = lat_col,id_col = id_col)
        out = add_nocturnal(df = out,target_zone_col = target_zone_col, nocturnal_col = 'day_night')
        out = add_high_seasson(df = out,datetime_col = datetime_col,high_season_col = 'high_season')
        weather = get_weather_data() # this will be replaced by an streaming API
        out = merge_weather_data(out, weather)
        out = process_weather_features(out)
        out = process_dt_minutes(out)
        out = add_initial_derived_parameters(out)
        out = propagation_speed_analysis(out)

        return out
