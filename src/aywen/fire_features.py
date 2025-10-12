import logging
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import richdem as rd
from pyproj import Transformer
from tqdm import tqdm
from aywen.utils import TimeOfDayFeatures
import rasterio
from rasterio.warp import transform as rio_transform
from pathlib import Path
import sys


try:
    from pyproj import datadir
except Exception:
    # pyproj not installed; nothing to do
    datadir = None

proj_dir = Path(sys.prefix) / "Library" / "share" / "proj"
if proj_dir.is_dir() and (proj_dir / "proj.db").exists():
    os.environ.setdefault("PROJ_DATA", str(proj_dir))
    os.environ.setdefault("PROJ_LIB",  str(proj_dir))
    if datadir is not None:
        datadir.set_data_dir(str(proj_dir))

logger = logging.getLogger("aywen_logger")

# quiet rasterio
for name in ("rasterio", "rasterio.env", "rasterio._env"):
  lg = logging.getLogger(name)
  lg.setLevel(logging.ERROR)   # or CRITICAL
  lg.propagate = False

# quiet fiona
for name in ("fiona", "fiona.env", "fiona._env"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    lg.propagate = False

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
    "initial_fuel",
    "diurnal_nocturnal",
    "high_season",
]

COVARIATES_CATEGORICAL = [
    "initial_fuel",
    "diurnal_nocturnal",
    "high_season",
]

PI_COVARIATES = [
    "diurnal_nocturnal",
    "high_season",
]

SPLIT_COLUMNS = ["split2", "split3"]

OTHERS = ["zone_alert", "hour", "kitral_fuel"]

DEFAULT_COLUMNS_DICT = {
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

DEFAULT_COLUMNS = ID_COLUMNS + GEOSPATIAL_COLUMNS + TIMESTAMP_COLUMNS + FACTORS + COVARIATES + TARGETS + OTHERS


CODE_KITRAL = {
    1: "Pastizales Mesomorficos Densos",
    2: "Pastizales Mesoomorficos Ralos",
    3: "Pastizales Higromorficos Densos",
    4: "Pastizales Higromorficos Ralos",
    5: "Chacareria Vinedos y Frutales",
    6: "Matorrales y Arbustos Mesomorficos Densos",
    7: "Matorrales y Arbustos Mesomorficos Medios y Ralos",
    8: "Matorrales y Arbustos Higromorficos Densos",
    9: "Matorrales y Arbustos Higromorificos Medios y Ralos",
    10: "Formaciones con predominancia de Chuesquea spp",
    11: "Formaciones con predominancia de Ulex spp",
    12: "Renovales Nativos diferentes al Tipo Siempreverde",
    13: "Renovales Nativos del Tipo Siempreverde",
    14: "Formaciones con predominancia de Alerzales",
    15: "Formaciones con predominancia de Araucaria",
    16: "Arbolado Nativo Denso",
    17: "Arbolado Nativo de Densidad Media",
    18: "Arbolado Nativo de Densidad Baja",
    19: "Plantaciones Coniferas Nuevas (0-3) sin Manejo",
    20: "Plantaciones Coniferas Jovenes (4-11) sin Manejo",
    21: "Plantaciones Coniferas Adultas (12-17) sin Manejo",
    22: "Plantaciones Coniferas Mayores (>17) sin Manejo",
    23: "Plantaciones Coniferas Jovenes (4-11) con Manejo",
    24: "Plantaciones Coniferas Adultas (12-17) con Manejo",
    25: "Plantaciones Coniferas Mayores (>17) con Manejo",
    26: "Plantaciones Eucaliptos Nuevas (0-3)",
    27: "Plantaciones Eucaliptos Jovenes (4-10)",
    28: "Plantaciones Eucalipto Adultas (>10)",
    29: "Plantaciones Latifoliadas y Mixtas",
    30: "Desechos Explotacion a Tala Rasa de Plantaciones",
    31: "Desechos Explotacion a Tala Rasa de Bosque Nativo",
    999: "No combustible",
}

DEFAULT_TO_KITRAL = {
    "Aserrin o Corteza en Acopio": "Desechos Explotacion a Tala Rasa de Plantaciones", #  made it up
    "Bosque nativo": "Arbolado Nativo Denso",
    "Desecho Agricola": "Matorrales y Arbustos Mesomorficos Medios y Ralos", # feedback from Ignacia
    "Desecho de Poda": "Desechos Explotacion a Tala Rasa de Plantaciones",
    "Desecho de Raleo": "Desechos Explotacion a Tala Rasa de Plantaciones",
    "Desecho de cosecha": "Desechos Explotacion a Tala Rasa de Plantaciones",
    "Desecho de cosecha nativo": "Desechos Explotacion a Tala Rasa de Bosque Nativo",
    "Estructurales, Casa, Bodega, etc": "No combustible",
    "Euca > 8 anos": "Plantaciones Eucalipto Adultas (>10)",
    "Euca de 0 - 3 anos": "Plantaciones Eucaliptos Nuevas (0-3)",
    "Euca de 4 - 7 anos": "Plantaciones Eucaliptos Jovenes (4-10)",
    "Maquinaria Pesada": "No combustible",
    "Matorral, Arbustos y O.Especies": "Matorrales y Arbustos Mesomorficos Densos",
    "Otros Combustibles No Vegetales": "No combustible",
    "Pastizal": "Pastizales Mesomorficos Densos",
    "Pi > de 14 anos con manejo": "Plantaciones Coniferas Adultas (12-17) con Manejo",
    "Pi > de 14 anos sin manejo": "Plantaciones Coniferas Adultas (12-17) sin Manejo",
    "Pi de 0 - 3 anos": "Plantaciones Coniferas Nuevas (0-3) sin Manejo",
    "Pi de 4 - 7 anos con manejo": "Plantaciones Coniferas Jovenes (4-11) con Manejo",
    "Pi de 4 - 7 anos sin manejo": "Plantaciones Coniferas Jovenes (4-11) sin Manejo",
    "Pi de 8 - 13 anos con manejo": "Plantaciones Coniferas Jovenes (4-11) con Manejo",
    "Pi de 8 - 13 anos sin manejo": "Plantaciones Coniferas Jovenes (4-11) sin Manejo",
    "Roce": "Desechos Explotacion a Tala Rasa de Plantaciones",
    "SIN INFORMACION": "No combustible",
    "Vehiculos de Carga": "No combustible",
    "Vehiculos de Personal": "No combustible",
}



# --------------- select columns ---------------

def select_columns_from_df(
    df: pd.DataFrame,
    include: list[str] = [],
    exclude: list[str] = [],
    strict: bool = False
) -> pd.DataFrame:

    logger.debug("called with df.shape=%s, include=%s, exclude=%s", df.shape, include, exclude)

    if not strict:
        include = [col for col in include if col in df.columns]
        exclude = [col for col in exclude if col in df.columns]

    out = df.copy()
    if include:
        missing = [col for col in include if col not in out.columns]
        if missing:
            raise KeyError(f"Columns not found in df: {missing}")
        include = [col for col in out.columns if col in include]  # preserve order
        out = out[include]
        logger.info("Selected columns: %s", include)

    if exclude:
        missing = [col for col in exclude if col not in out.columns]
        if missing:
            raise KeyError(f"Columns not found in df: {missing}")
        out = out.drop(columns=exclude)
        logger.info("Excluded columns: %s", exclude)

    return out


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
        shape_df = gpd.read_file(shapefile_path, engine="fiona")
        if shape_df.crs is None:
            shape_df = shape_df.set_crs(crs)   # assign (no reprojection)
        else:
            shape_df = shape_df.to_crs(crs)    # reproject
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
    or by sampling a DEM, then merge into `df` on `id_col`..

    Returns a new DataFrame with topography merged. Pre-existing topo columns
    ('elevation', 'slope_*', 'aspect_degrees') are dropped before merge to keep it idempotent.
    """

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, csv_path=%s, dem_path=%s",
        df.shape, csv_path, dem_path
    )

    from_csv = False
    from_dem = False

    if csv_path is not None:
        from_csv = True
    if dem_path is not None:
        from_dem = True

    if not (from_csv ^ from_dem):
        raise AssertionError("Exactly one of csv_path or dem_path must be provided.")

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
        logger.info("Loaded topographic table from %s", csv_path)
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

# --------------- period of day ---------------
def _get_period_day(row, datetime_col: str) -> str:
    # morning from 3:00 to 11:59
    # afternoon from 12:00 to 17:59
    # night from 18:00 to 2:59

    hour = pd.to_datetime(row[datetime_col]).hour
    if 3 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "night"

def add_period_day_to_df(
    df: pd.DataFrame,
    datetime_col: str = "start_datetime",
    period_col: str = "period_day"
    ) -> pd.DataFrame:

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, period_col=%s",
        df.shape, period_col
    )

    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in df.")

    # create a copy
    out = df.copy()
    out[period_col] = out.apply(lambda row: _get_period_day(row, datetime_col=datetime_col), axis=1)
    out[period_col] = pd.Categorical(out[period_col], categories=["morning", "afternoon", "night"], ordered=True)

    logger.info("Added %s.", period_col)

    return out

# --------------- diurnal/nocturnal ---------------

def _get_diurnal_nocturnal(row, datetime_col: str = "start_datetime") -> str:
    hour = pd.to_datetime(row[datetime_col]).hour
    if 3 <= hour < 18:
        return "diurnal"
    else:
        return "nocturnal"
    
def add_diurnal_nocturnal_to_df(
    df: pd.DataFrame,
    datetime_col: str = "start_datetime",
    diurnal_nocturnal_col: str = "diurnal_nocturnal"
    ) -> pd.DataFrame:

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, diurnal_nocturnal_col=%s",
        df.shape, diurnal_nocturnal_col
    )

    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in df.")

    # create a copy
    out = df.copy()
    out[diurnal_nocturnal_col] = out.apply(lambda row: _get_diurnal_nocturnal(row, datetime_col=datetime_col), axis=1)
    out[diurnal_nocturnal_col] = pd.Categorical(out[diurnal_nocturnal_col], categories=["diurnal", "nocturnal"], ordered=True)

    logger.info("Added %s.", diurnal_nocturnal_col)

    return out




# --------------- seasonal ---------------

def _is_high_season(ts):
    """
    Returns high or low depending if the timestamp falls between Dec 15 and Mar 15 (inclusive),
    ignoring the year.
    """
    m, d = ts.month, ts.day
    
    # Dec 15 to Mar 15
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


def add_fuel_from_tiff_to_df(
    df: pd.DataFrame,
    fuel_tiff_path: str,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    fuel_col: str = "fuel",
) -> pd.DataFrame:
    """
    Read a GeoTIFF of fuel types and map the raster value at each (lon, lat) in `df`
    into a new column `fuel_col`.

    Assumptions
    -----------
    - `df[lon_col]` and `df[lat_col]` are longitudes and latitudes in WGS84 (EPSG:4326).
    - The GeoTIFF is single-band (band 1 holds the fuel class/value).
    - Values outside the raster footprint or equal to the raster's NoData are set to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame with longitude/latitude columns.
    fuel_tiff_path : str
        Path or URL (supports COG over HTTP/S) to the fuel-type GeoTIFF.
    lon_col : str
        Name of the longitude column (degrees).
    lat_col : str
        Name of the latitude column (degrees).
    fuel_col : str
        Name of the output column to create/overwrite.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an added column `fuel_col`.
    """
    if lon_col not in df.columns or lat_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{lon_col}' and '{lat_col}' columns.")

    out = df.copy()

    # Adaptive batch size: keep ~20 batches, clamp to [10k, 100k].
    n = len(out)
    if n == 0:
        out[fuel_col] = np.array([], dtype=float)
        return out
    batch_size = min(100_000, max(10_000, n // 20))


    with rasterio.open(fuel_tiff_path) as src:
        raster_crs = src.crs
        nodata = src.nodata
        bounds = src.bounds

        # Input coords (WGS84)
        lons = out[lon_col].to_numpy(dtype=float, copy=False)
        lats = out[lat_col].to_numpy(dtype=float, copy=False)

        # Reproject points if needed (always_xy=True keeps (lon,lat) order)
        if raster_crs is None:
            raise ValueError("Raster has no CRS; cannot reproject points.")
        needs_reproj = str(raster_crs).upper() not in {"EPSG:4326", "OGC:CRS84"}
        if needs_reproj:
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)
            xs = np.asarray(xs)
            ys = np.asarray(ys)
        else:
            xs, ys = lons, lats

        # Fast bounds filter to skip sampling outside footprint
        in_bounds = (
            (xs >= bounds.left) &
            (xs <= bounds.right) &
            (ys >= bounds.bottom) &
            (ys <= bounds.top)
        )
        valid_idx = np.flatnonzero(in_bounds)

        # Prepare output; default NaN
        values = np.full(n, np.nan, dtype=float)

        if valid_idx.size == 0:
            out[fuel_col] = values
            return out

        # Batched sampling to keep memory and Python overhead in check
        # Use masked=True so we can map NoData/missing to NaN cleanly.
        band_index = 1
        for start in range(0, valid_idx.size, batch_size):
            end = min(start + batch_size, valid_idx.size)
            idx_batch = valid_idx[start:end]

            # Create a generator of (x,y) without building huge lists
            def coord_iter():
                xb = xs[idx_batch]
                yb = ys[idx_batch]
                # zip over numpy arrays yields tuples lazily
                return zip(xb, yb)

            # Stream samples; each is a masked array of shape (1,)
            # Avoid list() to keep peak memory low; write back as we go
            for offset, samp in enumerate(src.sample(coord_iter(), indexes=band_index, masked=True)):
                v = samp[0]  # masked scalar
                write_pos = idx_batch[offset]
                if getattr(v, "mask", False):
                    values[write_pos] = np.nan
                else:
                    val = float(v)
                    if nodata is not None and np.isfinite(val) and val == nodata:
                        values[write_pos] = np.nan
                    else:
                        values[write_pos] = val

            out[fuel_col] = values


    return out



def add_kitral_fuel_to_df(
        df: pd.DataFrame,
        fuel_tiff_path: str,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        fuel_col : str = "kitral_fuel",
) -> pd.DataFrame:
    """
    Add a fuel column to the DataFrame by reading from a GeoTIFF and mapping codes to descriptions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with longitude and latitude columns.
    fuel_tiff_path : str
        Path to the GeoTIFF file containing fuel codes.
    lon_col : str
        Name of the longitude column in df.
    lat_col : str
        Name of the latitude column in df.
    fuel_col : str
        Name of the output column to create with fuel descriptions.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an added column `fuel_col` containing fuel descriptions.
    """

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
        out = add_fuel_from_tiff_to_df(
            out,
            fuel_tiff_path,
            lon_col=lon_col,
            lat_col=lat_col,
            fuel_col=fuel_col,
        )
        logger.info("Added %s.", fuel_col)
    else:
        logger.info("Column %s already exists, skipping addition.", fuel_col)

    # Map codes to descriptions using CODE_KITRAL
    out[fuel_col] = out[fuel_col].map(CODE_KITRAL).fillna("Unknown")

    # Convert to categorical with all possible categories
    all_categories = list(CODE_KITRAL.values()) + ["Unknown"]
    out[fuel_col] = pd.Categorical(out[fuel_col], categories=all_categories)

    return out


def map_default_to_kitral(df: pd.DataFrame, fuel_col: str) -> pd.Series:
    """Map default fuel descriptions to Kitral codes."""

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, fuel_col=%s", df.shape, fuel_col)

    # create a copy
    out = df.copy()

    if fuel_col not in out.columns:
        raise KeyError(f"Column {fuel_col} not found in df.")
    
    # Map codes to descriptions using DEFAULT_FUEL_MAP
    out[fuel_col] = out[fuel_col].map(DEFAULT_TO_KITRAL)
    assert out[fuel_col].isnull().sum() == 0, "Some fuel codes could not be mapped."
    counts = out[fuel_col].value_counts()
    categories = counts[counts > 0].index
    out[fuel_col] = pd.Categorical(out[fuel_col], categories=categories)
    logger.info("Mapped %s to Kitral codes.", fuel_col)

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



def elliptical_propagation_speed(
        circular_speed, 
        lo_circular, 
        hi_circular, 
        ratio,
        prediction_major_axis_speed_mm_col,
        lo_major_axis_speed_mm_col,
        hi_major_axis_speed_mm_col,
        prediction_minor_axis_speed_mm_col,
        lo_minor_axis_speed_mm_col,
        hi_minor_axis_speed_mm_col):

    major_axis_speed = np.sqrt(ratio)*circular_speed
    lo_major = np.sqrt(ratio)*lo_circular
    hi_major = np.sqrt(ratio)*hi_circular
    lo_major = np.minimum(major_axis_speed, lo_major) # speed cannot be negative

    minor_axis_speed = circular_speed/np.sqrt(ratio)
    lo_minor = lo_major/ratio
    hi_minor = hi_major/ratio
    lo_minor = np.minimum(minor_axis_speed, lo_minor) # speed cannot be negative

    #return major_axis_speed, lo_major, hi_major, minor_axis_speed, lo_minor, hi_minor

    return {
        prediction_major_axis_speed_mm_col: major_axis_speed,
        lo_major_axis_speed_mm_col: lo_major,
        hi_major_axis_speed_mm_col: hi_major,
        prediction_minor_axis_speed_mm_col: minor_axis_speed,
        lo_minor_axis_speed_mm_col: lo_minor,
        hi_minor_axis_speed_mm_col: hi_minor
    }

def add_elliptical_propagation_speed_to_df(
    df: pd.DataFrame,
    circular_col: str = 'prediction_xgb',
    lo_circular_col: str = "lo_xgb",
    hi_circular_col: str = "hi_xgb",
    ratio: float = 3.0,
    prediction_major_axis_speed_mm_col: str = "prediction_major_axis_speed_mm",
    lo_major_axis_speed_mm_col: str = "lo_major_axis_speed_mm",
    hi_major_axis_speed_mm_col: str = "hi_major_axis_speed_mm",
    prediction_minor_axis_speed_mm_col: str = "prediction_minor_axis_speed_mm",
    lo_minor_axis_speed_mm_col: str = "lo_minor_axis_speed_mm",
    hi_minor_axis_speed_mm_col: str = "hi_minor_axis_speed_mm"
) -> pd.DataFrame:
    """
    Compute elliptical propagation speed and add it to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing circular propagation speed.
        circular_col (str): Column name for circular propagation speed (m/min).
        lo_circular_col (str): Column name for lower bound of circular speed.
        hi_circular_col (str): Column name for upper bound of circular speed.
        ratio (float): Ratio of major to minor axis speeds.
    Returns:
        pd.DataFrame: DataFrame with added elliptical propagation speed columns.
    """

    #--- Call args (debug) ---
    logger.debug("called with df.shape=%s, circular_col=%s, lo_circular_col=%s, hi_circular_col=%s, ratio=%s",
        df.shape, circular_col, lo_circular_col, hi_circular_col, ratio
    )

    # validate
    for col in [circular_col, lo_circular_col, hi_circular_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df.")

    # create a copy
    out = df.copy()
    columns_before = out.columns.tolist()
    out = out.join(out.apply(lambda row: pd.Series(elliptical_propagation_speed(
        row[circular_col], 
        row[lo_circular_col], 
        row[hi_circular_col], 
        ratio=ratio,
        prediction_major_axis_speed_mm_col=prediction_major_axis_speed_mm_col,
        lo_major_axis_speed_mm_col=lo_major_axis_speed_mm_col,
        hi_major_axis_speed_mm_col=hi_major_axis_speed_mm_col,
        prediction_minor_axis_speed_mm_col=prediction_minor_axis_speed_mm_col,
        lo_minor_axis_speed_mm_col=lo_minor_axis_speed_mm_col,
        hi_minor_axis_speed_mm_col=hi_minor_axis_speed_mm_col
    )), axis=1))
    new_cols = [c for c in out.columns if c not in columns_before]
    logger.info("Added %s", new_cols)
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
        topo_dem_path: str = None,
        fuel_tiff_path: str = None,
        kitral_fuel: bool = False,
        fuel_col = "initial_fuel",
        skip_weather: bool = False,
        skip_target: bool = False,
        include: list[str] = DEFAULT_COLUMNS,
        ) -> pd.DataFrame:

        logger.important("Starting feature engineering pipeline with df.shape=%s", df.shape)
        logger.important("Using fuel_col=%s (kitral_fuel=%s)", fuel_col, kitral_fuel)
        logger.important("Using include columns: %s", include)

        out = df.copy()
        out = add_zones_to_df(out, shapefile_path=zone_shapefile_path, crs=crs, lon_col=lon_col, lat_col=lat_col, zone_col=zone_col, new_zone_col="zone_alert", zone_threshold=zone_threshold)
        out = add_splitted_zones_to_df(out, zone_col="zone_alert", separation_keys=["zone_WE", "zone_NS"])
        out = add_sin_cos_hour_to_df(out, datetime_col=datetime_col, sin_col='sin_hour', cos_col='cos_hour')
        out = add_topo_to_df(out, csv_path=topo_csv_path, dem_path=topo_dem_path, lon_col=lon_col, lat_col=lat_col, id_col=id_col)
        out = add_diurnal_nocturnal_to_df(out, datetime_col=datetime_col, diurnal_nocturnal_col='diurnal_nocturnal')
        out = add_high_season_to_df(out, datetime_col = datetime_col,high_season_col = 'high_season')
        if kitral_fuel:# add kitral fuel from tiff
            out = add_kitral_fuel_to_df(out, fuel_tiff_path=fuel_tiff_path, lon_col=lon_col, lat_col=lat_col, fuel_col=fuel_col)
        else: # map default fuel to kitral codes
            out = map_default_to_kitral(out, fuel_col=fuel_col)
        if not skip_weather:
            weather = _get_weather_data() # this will be replaced by an streaming API
            out = _merge_weather_data(out, weather)
            out = add_weather_to_df(out)
        if not skip_target:
            out = add_dt_minutes_to_df(out, dt_column='dt_minutes')
            out = add_surface_to_df(out, radius_col='initial_radius_m')
            out = add_propagation_speed_to_df(out, radius_col='initial_radius_m', dt_col='dt_minutes', speed_col='propagation_speed_mm', add_kmh=True, speed_kmh_col='propagation_speed_kmh')
        out = select_columns_from_df(out, include=include)
        
        
        return out
