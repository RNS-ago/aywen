import logging
import math
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import richdem as rd
from pyproj import Transformer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from aywen.utils import TimeOfDayFeatures

logger = logging.getLogger("aywen_logger")

# --------------- miscellaneous ---------------

# constants.py

ID_COLUMNS = ["fire_id"]

GEOSPATIAL_COLUMNS = ["longitude", "latitude"]

TIMESTAMP_COLUMNS = ["start_datetime"]

FACTORS = ["zone_WE", "zone_NS"]

TARGETS = ["propagation_speed_mm", "dt_minutes", "initial_radius_m"]

COVARIATES = [
    "temperature",
    "wind_speed_kmh",
    "relative_humidity",
    "slope_degrees",
    "sin_hour",
    "cos_hour",
    "initial_fuel_reduced",
    "day_night",
    "high_season",
]

COVARIATES_CATEGORICAL = [
    "initial_fuel_reduced",
    "day_night",
    "high_season",
]

PI_COVARIATES = [
    "day_night",
    "high_season",
]

SPLIT_COLUMNS = ["split2", "split3"]

OTHERS = ["target_zone"]

DEFAULT_COLUMNS = {
    "id": ID_COLUMNS,
    "geospatial": GEOSPATIAL_COLUMNS,
    "timestamp": TIMESTAMP_COLUMNS,
    "factors": FACTORS,
    "covariates": COVARIATES,
    "covariates_categorical": COVARIATES_CATEGORICAL,
    "pi_covariates": PI_COVARIATES,
    "targets": TARGETS,
    "split": SPLIT_COLUMNS,
    "others": OTHERS,
}



# --------------- zone ---------------

def add_zones_to_df(df,
    shapefile_path: str,
    crs: str = "EPSG:4326",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    zone_col: str = "area",
    new_zone_col: str = "zone_alert",
    zone_threshold: int = 0,
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
    logger.debug("called with df.shape=%s, zone_col=%s", df.shape, zone_col)
    
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
    
    
    # create a copy
    out = df.copy()
    
    try:
        shape_df = gpd.read_file(shapefile_path)
        shape_df = shape_df.set_crs(crs=crs)
    except Exception:
        logger.exception("Failed to read shapefile.")
        raise

    pts = gpd.GeoDataFrame(
        data=out,
        geometry=gpd.points_from_xy(out[lon_col], out[lat_col]),
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
        logger.info("Row count changed after dropping NaNs: %d -> %d", len_gdf, len(gdf))
    gdf = gdf.rename(columns={zone_col: new_zone_col})
    gdf = gdf.drop(columns=["index_right", "geometry"])

    # drop factor with few observations
    if zone_threshold > 0:
        gdf = gdf[gdf[new_zone_col].map(gdf[new_zone_col].value_counts()) > zone_threshold]
        logger.info("Row count changed after dropping low-frequency zones: %d", len(gdf))

    if gdf.empty:
        logger.info("None of the coordinates were in the regions in the shapefile")
        return None
    else:
        logger.info("Added %s", new_zone_col)
        return gdf


def add_splitted_zones_to_df(
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
    
    # create a copy
    out = df.copy()
    
    #--- Validate input ---
    try:
        if isinstance(out, pd.DataFrame) == False:
            raise TypeError("df must be a pandas DataFrame.")
        if isinstance(zone_col, str) == False:
            raise TypeError("zone_col must be a string.")
        if isinstance(separation_keys, list) == False:
            raise TypeError("separation_keys must be a list.")
        if isinstance(separation_keys[0], str) == False:
            raise TypeError("separation_keys must be a list of strings.")
        if zone_col not in out.columns:
            raise ValueError(f"zone_col {zone_col} not in df.columns")
    except Exception:
        logger.exception("Invalid input to split_zones.")
        raise


    # --- Split zones ---
    logger.info("Splitting zones into separate columns.")

    # One-liner split with automatic column naming
    split_cols = out[zone_col].astype(str).str.split(" ", expand=True)

    # Rename and assign only the columns we need
    logger.info("Assigning zones to separate columns.")
    for i, key in enumerate(separation_keys):
        if i < split_cols.shape[1]:
            # Create categorical with unique values as categories
            out[key] = pd.Categorical(
                split_cols[i], categories=separation_levels[i], ordered=True
            )

    logger.info("Added %s and %s.", separation_keys[0], separation_keys[1])

    return out


# --------------- datetime ---------------
def add_sin_cos_hour_to_df(
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

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, datetime_col=%s, hour_col=%s, sin_col=%s, cos_col=%s, period=%s",
        df.shape, datetime_col, hour_col, sin_col, cos_col, period
    )

    logger.debug(
        "add_sin_cos_hour called with df.shape=%s, datetime_col=%s, period=%s",
        df.shape, datetime_col, period
    )

    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in df.")

    # Create a copy 
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

    logger.debug("Hour summary:\n%s", out[hour_col].describe())
    logger.info("Added %s and %s.", sin_col, cos_col)

    return out

# --------------- topography ---------------

def _dem_to_df(
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

    return topo_df


def add_topo_to_df(
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

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, from_csv=%s, from_dem=%s, csv_path=%s, dem_path=%s",
        df.shape, from_csv, from_dem, csv_path, dem_path
    )

    if not (from_csv ^ from_dem):
        raise AssertionError("Exactly one of from_csv or from_dem must be True.")

    if from_dem:
        if not dem_path:
            raise ValueError("dem_path must be provided when from_dem=True.")
        topo_df = _dem_to_df(
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
    logger.info("Added %s", keep_cols[1:])
    
    
    return out


# --------------- nocturnal ---------------

def _get_nocturnal(row, target_zone_col: str = "target_zone") -> str:
  if row[target_zone_col]=="NOCTURNO":
    return "Night"
  else:
    return "Day"


def add_nocturnal_to_df(
        df: pd.DataFrame,
        target_zone_col: str = "target_zone",
        nocturnal_col: str = "nocturnal",
        ) -> pd.DataFrame:
    """
    Add a 'nocturnal' categorical based on the 'target_zone' column.

    Returns a NEW DataFrame with the 'nocturnal' column.
    """

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, target_zone_col=%s, nocturnal_col=%s",
        df.shape, target_zone_col, nocturnal_col
    )

    if target_zone_col not in df.columns:
        raise KeyError(f"Column '{target_zone_col}' not found in df.")

    out = df.copy()
    out[nocturnal_col] = out.apply(lambda row: _get_nocturnal(row, target_zone_col=target_zone_col), axis=1)
    out[nocturnal_col] = pd.Categorical(out[nocturnal_col], categories=["Day", "Night"], ordered=True)

    logger.info("Added %s.", nocturnal_col)

    return out


# --------------- seasonal ---------------

def _is_high_season(ts):
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
    

def add_high_season_to_df(
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

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, datetime_col=%s, high_season_col=%s",
        df.shape, datetime_col, high_season_col
    )

    datetime_col = "start_datetime"
    high_season_col = "high_season"

    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in df.")

    out = df.copy()
    dt = pd.to_datetime(out[datetime_col])
    out[high_season_col] = dt.apply(_is_high_season)
    out[high_season_col] = pd.Categorical(out[high_season_col], categories=['low', 'high'], ordered=True)

    logger.info("Added %s.", high_season_col)

    return out


# --------------- fuel ---------------


def add_fuel_to_df(
        df: pd.DataFrame,
        fuel_tiff_path: str,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        fuel_col : str = "initial_fuel",
) -> pd.DataFrame:
    
    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, fuel_tiff_path=%s, lon_col=%s, lat_col=%s, fuel_col=%s",
        df.shape, fuel_tiff_path, lon_col, lat_col, fuel_col
    )
    
    # create a copy
    out = df.copy()

    if fuel_col in out.columns:
        logger.info("Column %s already exists in df, it will NOT be overwritten.", fuel_col)
        return out
    
    if not os.path.exists(fuel_tiff_path):
        raise FileNotFoundError(f"Fuel GeoTIFF file not found: {fuel_tiff_path}")

    if fuel_col not in out.columns:
        fuel_data = out[['longitude', 'latitude']].copy()
        fuel_data[fuel_col] = 'matorral__arbustos_y_oespecies'  # Placeholder for actual fuel data loading logic
        out = out.merge(fuel_data, on=["longitude", "latitude"], how="left")
        logger.info(f"Added {fuel_col}.")
    else:
        logger.info(f"Column {fuel_col} already exists, skipping addition.")

    out[fuel_col] = pd.Categorical(out[fuel_col], categories=out[fuel_col].unique())
    

    return out


def _old_add_fuel_reduced(
        df: pd.DataFrame,
        fuel_col: str = "initial_fuel",
       fuel_reduced_col: str = "initial_fuel_reduced",
       min_fuel_samples: int = 15
) -> pd.DataFrame:

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, fuel_col=%s, fuel_reduced_col=%s, min_fuel_samples=%s",
        df.shape, fuel_col, fuel_reduced_col, min_fuel_samples
    )

    # create a copy
    out = df.copy()

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
    logger.info("Added %s.", fuel_reduced_col)

    return out

def _reduce_fuel_types(s: pd.Series, min_fuel_samples) -> pd.Series:
    """Collapse rare levels in s to 'other'. Accepts int or fraction."""
    if min_fuel_samples < 1:
        n = s.size
        min_fuel_samples = max(1, math.ceil(min_fuel_samples * n))
        logger.debug("min_fuel_samples as fraction -> %d samples", min_fuel_samples)

    counts = s.value_counts()
    rare = counts[counts < min_fuel_samples].index
    logger.debug("Number of rare fuel types: %s", len(rare))
    return s.replace(rare, 'other')


def add_fuel_reduced(
        df: pd.DataFrame,
        factor1: str = "zone_WE",
        factor2: str = "zone_NS",
        fuel_col: str = "initial_fuel",
        fuel_reduced_col: str = "initial_fuel_reduced",
        min_fuel_samples: int = 15
) -> pd.DataFrame:
    """
    Reduce the number of categories in the fuel column by grouping rare categories into 'Other'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the fuel column.
        fuel_col (str): The name of the column containing the fuel types.
        fuel_reduced_col (str): The name of the new column to create with reduced fuel types.
        min_fuel_samples (int): Minimum number of samples for a category to be kept; otherwise grouped into 'Other'.
    
    Returns:
        pd.DataFrame: The DataFrame with the new reduced fuel column.
    """

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, fuel_col=%s, fuel_reduced_col=%s, min_fuel_samples=%s",
        df.shape, fuel_col, fuel_reduced_col, min_fuel_samples
    )

    # create a copy
    out = df.copy()

    out['tmp'] = (out[fuel_col]
       .astype(str)
       .str.replace('-', '', regex=False)
       .str.replace(' ', '_', regex=False)
       .str.replace(',', '_', regex=False)
       .str.replace('>', 'gt_', regex=False)
       .str.replace('<', 'lt_', regex=False)
       .str.replace('=', 'eq_', regex=False)
       .str.replace('.', '', regex=False)
       .str.lower()
    )

    # groupwise reduction (aligned back to original index)
    out[fuel_reduced_col] = out['tmp'].groupby([out[factor1], out[factor2]], observed=True).transform(lambda g: _reduce_fuel_types(g, min_fuel_samples=min_fuel_samples))
    out = out.drop(columns=['tmp'])
    out[fuel_reduced_col] = pd.Categorical(out[fuel_reduced_col], categories=out[fuel_reduced_col].unique())

    kept = out[fuel_reduced_col].nunique(dropna=True)
    original = df[fuel_col].nunique(dropna=True)
    logger.info("Fuel categories reduced: %d (from %d).", kept, original)
    logger.info("Added %s.", fuel_reduced_col)

    return out



# --------------- weather ---------------
def _get_weather_data(
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
    weather = pd.read_csv(path, sep=sep, usecols=usecols, low_memory=False)

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

def _merge_weather_data(
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

def add_weather_to_df(
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

    # --- Call args (debug) ---
    logger.debug("called with df.shape=%s, humidity_bounds=%s, wind_dir_bounds=%s, speed_kmh_threshold=%s, temp_c_threshold=%s, humidity_threshold=%s, drop_raw_wind=%s",
        df.shape, humidity_bounds, wind_dir_bounds, speed_kmh_threshold, temp_c_threshold, humidity_threshold, drop_raw_wind
    )

    # Create a copy
    out = df.copy()

    # Ensure numeric types if present
    for c in ["temperature", "relative_humidity", "wind_direction", "wind_speed_ms", "wind_speed_knots"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    created_cols = []

    # Clamp invalid ranges
    if "relative_humidity" in out.columns:
        lo, hi = humidity_bounds
        out["relative_humidity"] = out["relative_humidity"].where(
            (out["relative_humidity"] >= lo) & (out["relative_humidity"] <= hi), np.nan
        )
        created_cols.append("relative_humidity")

    if "wind_direction" in out.columns:
        lo, hi = wind_dir_bounds
        out.loc[(out["wind_direction"] < lo) | (out["wind_direction"] > hi), "wind_direction"] = np.nan
        created_cols.append("wind_direction")

    # Fill wind_speed_ms from knots if needed; then derive speeds
    has_ms = "wind_speed_ms" in out.columns
    has_knots = "wind_speed_knots" in out.columns

    if has_ms or has_knots:
        if not has_ms:
            out["wind_speed_ms"] = np.nan
            has_ms = True
            created_cols.append("wind_speed_ms")

        if has_knots:
            out["wind_speed_knots"] = pd.to_numeric(out["wind_speed_knots"], errors="coerce")
            out["wind_speed_ms"] = out["wind_speed_ms"].fillna(0.51445 * out["wind_speed_knots"])
            created_cols.append("wind_speed_knots")

        # Derived speeds (only if ms exists)
        out["wind_speed_mm"]  = out["wind_speed_ms"] * 60.0   # m/min
        out["wind_speed_kmh"] = out["wind_speed_ms"] * 3.6    # km/h
        created_cols.extend(["wind_speed_mm", "wind_speed_kmh"])

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
        created_cols.append("wind_speed_cat")
    else:
        out["wind_speed_cat"] = pd.Categorical([np.nan] * len(out))

    if "temperature" in out.columns:
        out["temperature_cat"] = pd.cut(
            out["temperature"],
            bins=[0, temp_c_threshold, np.inf],
            labels=["L", "H"],
            right=False,
        )
        created_cols.append("temperature_cat")
    else:
        out["temperature_cat"] = pd.Categorical([np.nan] * len(out))
        created_cols.append("temperature_cat")

    if "relative_humidity" in out.columns:
        out["humidity_cat"] = pd.cut(
            out["relative_humidity"],
            bins=[0, humidity_threshold, np.inf],
            labels=["D", "W"],
            right=False,
        )
        created_cols.append("humidity_cat")
    else:
        out["humidity_cat"] = pd.Categorical([np.nan] * len(out))
        created_cols.append("humidity_cat")

    # Fill missing cats with 'N'
    for c in ["wind_speed_cat", "temperature_cat", "humidity_cat"]:
        out[c] = out[c].cat.add_categories("N").fillna("N")

    # Combined indices
    out["weather_index"] = np.where(
        (out["wind_speed_cat"] != "N") & (out["temperature_cat"] != "N"),
        out["wind_speed_cat"].astype(str) + out["temperature_cat"].astype(str),
        "NA",
    )
    created_cols.append("weather_index")

    out["weather_index_full"] = np.where(
        (out["wind_speed_cat"] != "N")
        & (out["temperature_cat"] != "N")
        & (out["humidity_cat"] != "N"),
        out["wind_speed_cat"].astype(str)
        + out["temperature_cat"].astype(str)
        + out["humidity_cat"].astype(str),
        "NA",
    )

    logger.info("Added %s", created_cols)

    return out


# --------------- target / propagation ---------------

def add_dt_minutes_to_df(
        df: pd.DataFrame, 
        dt_column: str = 'dt_minutes', 
        start_time_column: str = 'start_datetime', 
        arrival_time_column: str = 'arrival_datetime_inc'
        ) -> pd.DataFrame:
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
        - dt_column: numeric difference in minutes
        - f"{dt_column}_cat": categorical bins (0-10, 10-30, 30+)
    """
    logger.debug("called with df.shape=%s, dt_column=%s, start_time_column=%s, arrival_time_column=%s", df.shape, dt_column, start_time_column, arrival_time_column)

    # validate
    if start_time_column not in df.columns or arrival_time_column not in df.columns:
        raise KeyError(f"Both {start_time_column} and {arrival_time_column} must exist in df.columns")

    out = df.copy()
    out[start_time_column] = pd.to_datetime(out[start_time_column])
    out[arrival_time_column] = pd.to_datetime(out[arrival_time_column])

    out[dt_column] = (out[arrival_time_column] - out[start_time_column]).dt.total_seconds() / 60.0

    # add categorical bins
    bins = [0, 10, 30, np.inf]
    labels = ['0-10', '10-30', '30+']
    dt_column_cat = f"{dt_column}_cat"
    out[dt_column_cat] = pd.cut(out[dt_column], bins=bins, labels=labels, right=False)

    logger.debug("Summary for %s:\n%s", dt_column, out[dt_column].describe())
    logger.info("Added %s and %s_cat.", dt_column, dt_column_cat)

    return out

def add_surface_to_df(
    df: pd.DataFrame,
    surface_ha_col: str = "initial_surface_ha",
    m2_col: str = "initial_surface_m2",
    radius_col: str = "initial_radius_m",
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
    m2_col : str
        Name of output area column in square meters.
    radius_col : str
        Name of output radius column (meters).
    outlier_quantile : float
        Upper quantile used to trim extreme m² outliers (default 0.99).

    Returns
    -------
    pd.DataFrame
        Copy of df with new columns and outliers removed.
    """

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, surface_ha_col=%s, m2_col=%s, radius_col=%s, outlier_quantile=%s",
        df.shape, surface_ha_col, m2_col, radius_col, outlier_quantile
    )


    if surface_ha_col not in df.columns:
        raise KeyError(f"Column '{surface_ha_col}' not found in df.")

    # Create a copy
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

    logger.info("Added %s and %s.", m2_col, radius_col)

    return out


def add_propagation_speed_to_df(
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

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, radius_col=%s, dt_col=%s, speed_col=%s, add_kmh=%s, speed_kmh_col=%s, lower_limit=%s, upper_limit=%s, return_summary=%s",
        df.shape, radius_col, dt_col, speed_col, add_kmh, speed_kmh_col, lower_limit, upper_limit, return_summary
    )

    # validate
    if radius_col not in df.columns:
        raise KeyError(f"Column '{radius_col}' not found in df.")
    if dt_col not in df.columns:
        raise KeyError(f"Column '{dt_col}' not found in df.")

    # create a copy
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

        logger.info("Added %s and %s.", speed_col, speed_kmh_col)

    return (out, summary) if return_summary else out


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



# --------------- full pipeline ---------------



def feature_engineering_pipeline(
        df: pd.DataFrame,
        id_col: str = "fire_id",
        datetime_col: str = "start_datetime",
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        crs: str = "EPSG:4326",
        zone_shapefile_path: str = None,
        zone_col: str = "area",
        zone_threshold: int = 5,
        topo_csv_path: str = None,
        target_zone_col: str = "target_zone",
        fuel_tiff_path: str = None,
        fuel_col: str = "initial_fuel",
        skip_fuel_reduced: bool = False,
        skip_train_test_split: bool = False,
        skip_weather: bool = False,
        skip_target: bool = False,
        ) -> pd.DataFrame:

        out = df.copy()
        out = add_zones_to_df(out, shapefile_path=zone_shapefile_path, crs=crs, lon_col=lon_col, lat_col=lat_col, zone_col=zone_col, new_zone_col="zone_alert", zone_threshold=zone_threshold)
        out = add_splitted_zones_to_df(out, zone_col="zone_alert", separation_keys=["zone_WE", "zone_NS"])
        out = add_sin_cos_hour_to_df(out, datetime_col=datetime_col, sin_col='sin_hour', cos_col='cos_hour')
        out = add_topo_to_df(out, csv_path=topo_csv_path, lon_col=lon_col, lat_col=lat_col, id_col=id_col)
        out = add_nocturnal_to_df(out, target_zone_col=target_zone_col, nocturnal_col='day_night')
        out = add_high_season_to_df(out, datetime_col = datetime_col,high_season_col = 'high_season')
        out = add_fuel_to_df(df = out, fuel_col = fuel_col, fuel_tiff_path=fuel_tiff_path)
        if not skip_fuel_reduced:
            out = add_fuel_reduced(df = out, fuel_col = fuel_col, fuel_reduced_col = 'initial_fuel_reduced')
        if not skip_train_test_split:
            out = add_train_test_split_to_df(out, zone_col="zone_alert")
        if not skip_weather:
            weather = _get_weather_data() # this will be replaced by an streaming API
            out = _merge_weather_data(out, weather)
            out = add_weather_to_df(out)
        if not skip_target:
            out = add_dt_minutes_to_df(out, dt_column='dt_minutes')
            out = add_surface_to_df(out, radius_col='initial_radius_m')
            out = add_propagation_speed_to_df(out, radius_col='initial_radius_m', dt_col='dt_minutes', speed_col='propagation_speed_mm', add_kmh=True, speed_kmh_col='propagation_speed_kmh')

        return out
