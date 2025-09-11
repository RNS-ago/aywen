import glob
import logging
import numpy as np
import os
import pandas as pd
import re

from .utils import log_step, concatnate_name_from_paths


logger = logging.getLogger("aywen_logger")


def load_datasets(
    dataset_path: list[str] | str, 
    separator: str = ";", 
    low_memory: bool = False
) -> list[pd.DataFrame]:
    """
    Load datasets from CSV files.

    Parameters:
        dataset_path (list[str] | str): Directory path, list of paths, or a single path to fire data CSV files.
        separator (str): CSV delimiter. Default is ';'.
        low_memory (bool): Whether to allow pandas to internally process chunks. Default is False.

    Returns:
        list[pd.DataFrame]: A list of pandas DataFrames.
    """
    
    # --- Call args (debug) ---
    logger.debug("called with dataset_path=%s, separator=%s, low_memory=%s", dataset_path, separator, low_memory)

    # --- Validate input ---
    try:
        if isinstance(dataset_path, list):
            if not all(isinstance(p, str) for p in dataset_path):
                raise TypeError("All elements of dataset_path must be strings.")
        elif isinstance(dataset_path, str):
            if dataset_path.endswith(".csv"):
                dataset_path = [dataset_path]
            else:
                dataset_path = glob.glob(f"{dataset_path}/*.csv")
        else:
            raise TypeError("dataset_path must be a string or a list of strings.")

        if not dataset_path:
            raise FileNotFoundError("No CSV files found in the given path(s).")
    except Exception:
        logger.exception("\tFailed while validating/expanding dataset_path.")
        raise

    # --- Iterate and load ---
    logger.info("Loading datasets from %s.", dataset_path)
    dfs: list[pd.DataFrame] = []
    for path in dataset_path:
        if not os.path.isfile(path):
            logger.error("\tCSV file not found: %s", path)
            raise FileNotFoundError(f"CSV file not found: {path}")

        dataset_name = os.path.splitext(os.path.basename(path))[0]
        logger.info("\tReading CSV: %s", path)

        try:
            df = pd.read_csv(path, sep=separator, low_memory=low_memory)
        except Exception:
            logger.exception("\tFailed to read CSV %s", path)
            raise

        try:
            # Normalize columns and attach metadata
            df.columns = (
                df.columns
                .str.strip()
                .str.replace(r"\s+", "_", regex=True)
                .str.upper()
            )
            df.attrs["name"] = dataset_name
            df.loc[:, "ARCHIVO"] = dataset_name
            before = len(df)
            df = df.dropna(how="all")
            after = len(df)
            if after < before:
                logger.warning(
                    "Dropped %s all-NaN rows in %s", before - after, dataset_name
                )
        except Exception:
            logger.exception("\tFailed while post-processing DataFrame from %s", path)
            raise

        dfs.append(df)
        logger.info(
            "\tLoaded %s: %s rows, %s columns",
            dataset_name, len(df), len(df.columns)
        )

    logger.info("Successfully loaded %s dataset(s).", len(dfs))
    return dfs


def check_dataset_compatibility(    
    dfs: list[pd.DataFrame], 
    check_matching_columns: bool = False,
    check_unique_rows: bool = False, 
    unique_rows_key: str = "FICHA"
) -> bool:
    """
    Check if multiple DataFrames are compatible:
    - All have the same columns
    - Optionally, no overlapping values in a given key column

    Parameters:
        dfs (list[pd.DataFrame]) : List of DataFrames to compare.
        check_unique_rows (bool): If True, ensures `unique_rows_key` values are unique across DataFrames.
        unique_rows_key (str): Column name used for uniqueness checking.

    Returns:
        bool: True if compatible, raises ValueError otherwise.
    """

    logger.info("Checking %s dataset list compatibility.", [df.attrs["name"] for df in dfs])

    # --- Call args (debug) ---
    logger.debug(
        "called: n_dfs=%s, check_matching_columns=%s, check_unique_rows=%s, unique_rows_key=%s",
        None if dfs is None else len(dfs),
        check_matching_columns,
        check_unique_rows,
        unique_rows_key,
    )

    # --- Validate input ---
    try:
        if dfs is None:
            raise ValueError("dfs cannot be None. Expected a list of DataFrames.")
        if not isinstance(dfs, list):
            raise TypeError(f"dfs must be a list of DataFrames, got {type(dfs).__name__}.")
        if len(dfs) == 0:
            raise ValueError("dfs is an empty list. Provide at least one DataFrame.")
        if not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise TypeError("All elements of `dfs` must be pandas DataFrames.")
    except Exception:
        logger.exception("\tInvalid input to check_dataset_compatibility.")
        raise

    # --- Column consistency baseline ---
    reference_columns = set(dfs[0].columns)
    logger.info("\tReference DataFrame has %d columns.", len(reference_columns))
    logger.debug("\tReference columns: %s", sorted(reference_columns))

    if check_unique_rows:
        if unique_rows_key not in reference_columns:
            msg = f"Column '{unique_rows_key}' not found in the first DataFrame."
            logger.error("\t" + msg)
            raise KeyError(msg)
        # Use dropna + unique to avoid NaN pollution
        first_unique = pd.Series(dfs[0][unique_rows_key]).dropna().unique()
        unique_rows = set(first_unique)
        logger.info(
            "\tInitialized unique set with %d values from df[0] for key '%s'.",
            len(unique_rows), unique_rows_key
        )
    else:
        unique_rows = set()

    # --- Iterate and check ---
    for i, df in enumerate(dfs[1:], start=1):
        if check_matching_columns:
            current_cols = set(df.columns)
            if current_cols != reference_columns:
                missing = reference_columns - current_cols
                extra = current_cols - reference_columns
                logger.error(
                    "\tColumn mismatch at df[%d]: missing=%s, extra=%s",
                    i, sorted(missing), sorted(extra)
                )
                raise ValueError(
                    f"DataFrame at index {i} has different columns. "
                    f"Missing: {sorted(missing)}; Extra: {sorted(extra)}"
                )
            logger.info("\tdf[%d] columns match reference.", i)

        if check_unique_rows:
            if unique_rows_key not in df.columns:
                msg = f"Column '{unique_rows_key}' not found in DataFrame at index {i}."
                logger.error("\t" + msg)
                raise KeyError(msg)

            df_unique_rows = set(pd.Series(df[unique_rows_key]).dropna().unique())
            duplicates = df_unique_rows & unique_rows
            if duplicates:
                # Avoid logging huge sets
                sample = list(sorted(duplicates))[:10]
                logger.error(
                    "\tDuplicate '%s' values found at df[%d]: count=%d, sample=%s",
                    unique_rows_key, i, len(duplicates), sample
                )
                raise ValueError(
                    f"DataFrame at index {i} has duplicate {unique_rows_key} values "
                    f"already present in previous DataFrames (e.g., {sample}, total {len(duplicates)})."
                )
            unique_rows |= df_unique_rows
            logger.info(
                "\tdf[%d] unique accumulation for '%s': total=%d",
                i, unique_rows_key, len(unique_rows)
            )

    logger.info("Datasets are compatible across %d DataFrame(s).", len(dfs))
    return True


def convert_and_add_date_time_columns(df: pd.DataFrame, date_column: str = "INICIO") -> pd.DataFrame:
    """
    Convert a column in a pandas DataFrame to datetime format.

    Parameters:
        df (pd.DataFrame): The pandas DataFrame to convert.
        date_column (str): The name of the column to convert.
    
    Returns:
        pd.DataFrame: The converted pandas DataFrame.
    """
    logger.info("Converting %s to datetime in %s.", date_column, df.attrs["name"])
    
    # --- Call args (debug) ---
    logger.debug("called with df=%s, date_column=%s", df.attrs["name"], date_column)
    
    # --- Validate input ---
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
        if date_column not in df.columns:
            raise KeyError(f"Column '{date_column}' not found in df.")
    except Exception:
        logger.exception("\tInvalid input to convert_to_datetime.")
        raise
    
    # --- Convert ---
    try:
        df.loc[:, date_column] = pd.to_datetime(df[date_column].astype(str).str.strip(), format="%Y-%m-%d %H:%M", errors="coerce")
        df[date_column] = df[date_column].astype('datetime64[ns]')
        df[f"DIA_{date_column}"] = df[date_column].dt.date
        df[f"HORA_{date_column}"] = df[date_column].dt.hour
    except Exception:
        logger.exception("\tFailed to convert %s to datetime.", date_column)
        raise
    
    logger.info("Successfully converted %s to datetime.", date_column)
    return df


def standardize_dispatch_reports_ID_and_fliter(
    df: pd.DataFrame,
    report_type_column: str = "TIPO_FICHA",
    selected_report_types: list[str] = ["FOCO"],
    filter_report_types: bool = True,
    standard_fire_dispatch_id_column: str = "ID_INCENDIO_DESPACHO",
    dispatch_id_column: str = "ID_DESPACHO",
    fire_id_column: str = "ID_FOCO_INCENDIO",
    fire_id_length: int = 9,
    zone_column: str = "ZONA",
    selected_zones: list[str] = ["ZONA CHILLAN", "ZONA CONSTITUCION", "ZONA ARAUCO", "ZONA VALDIVIA"],
    filter_zones: bool = True,
    zone_name_prefix: str = "ZONA ",
    remove_zone_prefix: bool = True,
) -> pd.DataFrame:
    """"
    Filter a DataFrame to only dispatches with a given report type and standardize the IDs.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        report_type_column (str): The name of the column to filter on.
        filter_report_types (bool): Whether to filter on report types.
        selected_report_types (list[str]): The list of report types to filter on.
        standard_fire_dispatch_id_column (str): The name of the newly standardized column.
        id_length (int): The length of the standardized IDs.
        zone_column (str): The name of the column to filter on.
        selected_zones (list[str]): The list of zones to filter on.
        filter_zones (bool): Whether to filter on zones.
        zone_name_prefix (str): The prefix to remove from zone names.
        remove_zone_prefix (bool): Whether to remove the prefix from zone names.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # --- Call args (debug) ---
    logger.debug(
        """called with df=%s, report_type_column=%s, filter_report_types=%s,
        selected_report_types=%s, standard_fire_dispatch_id_column=%s,
        dispatch_id_column=%s, fire_id_column=%s, fire_id_length=%s, zone_column=%s,
        selected_zones=%s, filter_zones=%s, zone_name_prefix=%s, remove_zone_prefix=%s""",
        df.attrs["name"],
        report_type_column,
        filter_report_types,
        selected_report_types,
        standard_fire_dispatch_id_column,
        dispatch_id_column,
        fire_id_column,
        fire_id_length,
        zone_column,
        selected_zones,
        filter_zones,
        zone_name_prefix,
        remove_zone_prefix,
    )
    
    
    # --- Validate input ---
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
        if report_type_column not in df.columns:
            raise KeyError(f"Column '{report_type_column}' not found in df.")
        if zone_column not in df.columns:
            raise KeyError(f"Column '{zone_column}' not found in df.")
    except Exception:
        logger.exception("Invalid input to filter_dispatchs.")
        raise
    
    # --- Filter ---
    try:
        df = df.copy()
        # Filter by report type
        if filter_report_types:
            df = df[df[report_type_column].isin(selected_report_types)]

        # Standardize the fire dispatch ID
        df.loc[:, standard_fire_dispatch_id_column] = df[fire_id_column].astype(str) + df[dispatch_id_column].astype(str)
        df = df[df[fire_id_column].astype(str).str.len() == fire_id_length]
        
        # Filter only unique dispatchs, keeping the first one
        df = df.drop_duplicates(subset=[standard_fire_dispatch_id_column], keep="first")

        # Filter only dispatches in the zones we are interested in and remove the "ZONA " prefix
        if filter_zones:
            df = df[df[zone_column].isin(selected_zones)]
        if remove_zone_prefix:
            df.loc[:, zone_column] = df[zone_column].str.replace(zone_name_prefix, "")
        
    except Exception:
        logger.exception("Failed to filter and/or standardize dispatches.")
        raise
    
    logger.debug("df has %d unique dispatches.", len(df))

    return df


def standardize_fire_reports_IDs(
    df: pd.DataFrame,
    standard_fire_id_column: str = "ID_FOCO_INCENDIO",
    fire_id_column: str = "FICHA",
    id_length: int = 9,
) -> pd.DataFrame:
    """
    Standardize the fire ID column and filter only fire IDs with a length of 9.
    Parameters:
        df (pd.DataFrame): The pandas DataFrame to standardize.
        standard_fire_id_column (str): The name of the newly standardized column.
        fire_id_column (str): The name of the column to standardize.

    Returns:
        pd.DataFrame: The standardized pandas DataFrame.
    """
    # --- Call args (debug) ---
    logger.debug("called with df=%s, standard_fire_id_column=%s, fire_id_column=%s", df.attrs["name"], standard_fire_id_column, fire_id_column)
    
    # --- Validate input ---
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
        if fire_id_column not in df.columns:
            raise KeyError(f"Column '{fire_id_column}' not found in df.")
    except Exception:
        logger.exception("Invalid input to standardize_fire_id.")
        raise
    
    # --- Standardize ---
    try:
        if standard_fire_id_column not in df.columns:
            df = df.rename(columns={fire_id_column: standard_fire_id_column})
        df = df[df[standard_fire_id_column].astype(str).str.len() == id_length]
    except Exception:
        logger.exception("Failed to standardize fire IDs.")
        raise
    
    logger.debug("df has %d unique fire IDs.", len(df))
    return df


def _normalize_glosa_df1(s: pd.Series) -> pd.Series:
    """
    df1 rules
    - HELO:  H<digits>      -> BHA<digits>
    - Aircraft: A<digits>? -> AA<digits> 
    - Brigades: B<digits>   -> BA<digits>
    """
    
    
    out = s.astype(str)

    # HELO: H123 -> BHA123
    out = out.str.replace(r'^H(\d+)\b', r'BHA\1', regex=True)

    # Aircraft: A123 -> AA123
    out = out.str.replace(r'^A(\d+)\b', r'AA\1', regex=True)

    # Brigades: B123 -> BA123
    out = out.str.replace(r'^B(\d+)\b', r'BA\1', regex=True)

    return out

def _normalize_glosa_df2(s: pd.Series) -> pd.Series:
    """
    df2 rules
    - HELO already as BHA… : strip spaces/hyphens anywhere in value, but only when it starts with BHA
      e.g., 'BHA-12 3 extra' -> 'BHA123'
    - Aircrafts: AA<digits> keep only that token (drop trailing text) 
      e.g., 'AA123 - foo' -> 'AA123'
    - Brigades: BA<digits> keep only that token (drop trailing text)
      e.g., 'BA123 something' -> 'BA123'
    """
    
    out = s.astype(str)

    # HELO: BHA… keep only first token, remove - or spaces
    out = out.str.replace(r'^(BHA[\d\s-]+).*$', r'\1', regex=True)
    out = out.str.replace(r'[-\s]', '', regex=True)

    # Aircraft: AA… keep only first token
    out = out.str.replace(r'^(AA\d+).*$', r'\1', regex=True)

    # Brigades: BA… keep only first token
    out = out.str.replace(r'^(BA\d+).*$', r'\1', regex=True)

    return out

def standardize_response_teams(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize GLOSA in df1 and df2 dispatch DataFrames.
    """
    df1_norm = df1.copy()
    df2_norm = df2.copy()

    df1_norm.loc[:, 'GLOSA'] = _normalize_glosa_df1(df1_norm['GLOSA'])
    df2_norm.loc[:, 'GLOSA'] = _normalize_glosa_df2(df2_norm['GLOSA'])

    return df1_norm, df2_norm


def match_fires_and_dispatches(fire_df: pd.DataFrame, dispatch_df: pd.DataFrame, fire_id_column: str = "ID_FOCO_INCENDIO") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match fire and dispatch DataFrames based on their IDs.
    
    Parameters:
        fire_df (pd.DataFrame): The fire DataFrame to match.
        dispatch_df (pd.DataFrame): The dispatch DataFrame to match.
        fire_id_column (str): The name of the column with the fire IDs to match.
        
    Returns:
        pd.DataFrame: The matched pandas DataFrame.
    """
    
    fire_report_IDs = fire_df[fire_id_column].unique()
    dispatch_report_fire_IDs = dispatch_df[fire_id_column].unique()
    
    fire_report_with_dispatch = fire_df[fire_df[fire_id_column].isin(dispatch_report_fire_IDs)]
    fire_report_without_dispatch = fire_df[~fire_df[fire_id_column].isin(dispatch_report_fire_IDs)]
    
    dispatch_report_with_fire = dispatch_df[dispatch_df[fire_id_column].isin(fire_report_IDs)]
    dispatch_report_without_fire = dispatch_df[~dispatch_df[fire_id_column].isin(fire_report_IDs)]
    
    #TODO: Turn into logging
    #print(f"Unique fire reports: {len(fire_report_IDs)}\n"
    #      f"Unique dispatch reports: {len(dispatch_report_fire_IDs)}\n"
    #      f"Fire reports with matching dispatches: {len(fire_report_with_dispatch)}\n"
    #      f"Fire reports without matching dispatches: {len(fire_report_without_dispatch)}\n"
    #      f"Dispatch reports with matching fires: {len(dispatch_report_with_fire)}\n"
    #      f"Dispatch reports without matching fires: {len(dispatch_report_without_fire)}\n"
    #      )
    
    return fire_report_with_dispatch, dispatch_report_with_fire


def fire_summary_by_season(
    df: pd.DataFrame,
    dfd: pd.DataFrame,
    key: str = "ID_FOCO_INCENDIO",
    season_col: str = "TEMPORADA",
    archivo_col: str = "ARCHIVO",
) -> pd.DataFrame:
    """
    Build a per-season summary matching your original logic.

    - For each TEMPORADA in df:
        * Take the unique incident IDs (key) present in df for that season.
        * Filter dfd to those IDs to find which have a 'foco' (same key).
        * Count incidents with/without foco in df for that season.
    - 'Archivo' mirrors your code: the first value found in the group (if any).

    Parameters
    ----------
    df : pd.DataFrame
        Main dataframe with columns [season_col, key, archivo_col].
    dfd : pd.DataFrame
        Secondary dataframe to check presence of key (the “foco” source).
    key : str
        Column name for the ID used to match.
    season_col : str
        Column name for the season grouping (TEMPORADA).
    archivo_col : str
        Column from which to take the first value per season (ARCHIVO).

    Returns
    -------
    pd.DataFrame
        One row per season with counts and optional extrema.
    """
    # Defensive copies not required as we don't mutate, but we coerce views to Series safely.
    if key not in df.columns or key not in dfd.columns:
        raise KeyError(f"Both dataframes must contain the '{key}' column.")
    if season_col not in df.columns:
        raise KeyError(f"`df` must contain the '{season_col}' column.")
    if archivo_col not in df.columns:
        raise KeyError(f"`df` must contain the '{archivo_col}' column.")

    summary_rows = []

    for temporada, dfi_temp in df.groupby(season_col, dropna=False):
        # Preserve your original "first ARCHIVO" behavior, guard for empty groups
        archivo_val = dfi_temp[archivo_col].iloc[0] if len(dfi_temp) else None

        fichas_temp = pd.Series(dfi_temp[key]).dropna().unique()
        if len(fichas_temp) == 0:
            # No IDs in this season; all rows counted but none with foco
            dfi_con_foco = dfi_temp.iloc[0:0]
            dfi_sin_foco = dfi_temp
            dfd_temp = dfd.iloc[0:0]
        else:
            dfd_temp = dfd[dfd[key].isin(fichas_temp)]
            focos_temp = pd.Series(dfd_temp[key]).dropna().unique()
            dfi_con_foco = dfi_temp[dfi_temp[key].isin(focos_temp)]
            dfi_sin_foco = dfi_temp[~dfi_temp[key].isin(focos_temp)]

        row = {
            "Archivo": archivo_val,
            "Temporada": temporada,
            "Incendios": int(len(dfi_temp)),
            "Incendios con Id-Foco": int(len(dfi_con_foco)),
            "Incendios sin Id-Foco": int(len(dfi_sin_foco)),
        }

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # Optional nice ordering
    cols = ["Archivo", "Temporada", "Incendios", "Incendios con Id-Foco", "Incendios sin Id-Foco"]
    df_summary = df_summary[cols]

    # Sort by Temporada if sortable; otherwise leave as encountered order
    with pd.option_context('mode.use_inf_as_na', True):
        try:
            df_summary = df_summary.sort_values(by=["Temporada"])
        except Exception:
            pass

    return df_summary


def selecting_and_renaming_fire_variables(
    fire_df: pd.DataFrame, dispatch_df: pd.DataFrame, variables: list[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:

    key_id = "ID_FOCO_INCENDIO"

    # TODO: Tomar las variable TEMPORAL de la planilla Despachos pues la de incendios tiene errores
    geotemporal = {
        "GEO": ["LATITUD", "LONGITUD", "REGION", "PROVINCIA", "COMUNA", "PREDIO"],
        "TEMPORAL": ["TEMPORADA","DIA_INICIO", "HORA_INICIO", "INICIO", "ARRIBO", "CONTROL", "EXTINCION"],
    }

    complejidad = {
        "SITUACIONAL": ["ZONA_META", "COMBUSTIBLE_INICIAL", "PIRA", "EUCA", "OTRAS"],
        "CAUSA": ["GRUPO_CAUSAL", "CAUSA"],
    }

    desempeno = {"SUPERFICIE": ["SUP_INICIAL", "SUP", "CLASE", "GASTO_COMBATE_UF"]}

    # building up the final data frame
    fire_df = fire_df[
        [key_id]
        + geotemporal["GEO"]
        + geotemporal["TEMPORAL"]
        + complejidad["SITUACIONAL"]
        + complejidad["CAUSA"]
        + desempeno["SUPERFICIE"]
        + ["ARCHIVO"]
    ].copy()

    rename_dict = {
        "ID_FOCO_INCENDIO": "fire_id",
        "LATITUD": "latitude",
        "LONGITUD": "longitude",
        "REGION": "region",
        "PROVINCIA": "province",
        "COMUNA": "commune",
        "PREDIO": "property",
        "TEMPORADA": "season",
        "DIA_INICIO": "start_day",
        "HORA_INICIO": "start_time",
        "INICIO": "start_datetime",
        "ARRIBO": "arrival_datetime_inc",  # will be compared with information from despachos
        "CONTROL": "control_datetime",
        "EXTINCION": "extinction_datetime",
        "ZONA_META": "target_zone",
        "COMBUSTIBLE_INICIAL": "initial_fuel",
        "PIRA": "pyre",
        "EUCA": "euca",
        "OTRAS": "other_fuels",
        "GRUPO_CAUSAL": "causal_group",
        "CAUSA": "cause",
        "SUP_INICIAL": "initial_surface_ha",
        "SUP": "final_surface_ha",
        "CLASE": "class",
        "GASTO_COMBATE_UF": "combat_cost_per_unit",
        "ARCHIVO": "file_source",
    }

    fire_df = fire_df.rename(columns=rename_dict)

    dispatch_df = dispatch_df.rename(
        columns={
            "ID_FOCO_INCENDIO": "fire_id",
            "ZONA": "zone",
            "ARCHIVO": "file_source",
        }
    )

    dispatch_df.columns = dispatch_df.columns.str.lower()

    return fire_df, dispatch_df


def map_zones_from_dispatch_to_fires(
    fire_df: pd.DataFrame,
    dispatch_df: pd.DataFrame,
    zone_column: str = "zone",
    fire_id_column: str = "fire_id",
) -> pd.DataFrame:
    """
    Map zones from dispatch to fires.

    Parameters:
        fire_df (pd.DataFrame): The DataFrame containing the fire data.
        dispatch_df (pd.DataFrame): The DataFrame containing the dispatch data.
        zone_column (str): The name of the column containing the zones in the dispatch DataFrame.
        fire_id_column (str): The name of the column containing the fire IDs in the fire DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the zones mapped to the fire IDs.
    """
    # --- Call args (debug) ---
    logger.debug("called with fire_df.shape=%s, dispatch_df.shape=%s, zone_column=%s, fire_id_column=%s", fire_df.shape, dispatch_df.shape, zone_column, fire_id_column)

    # --- Validate input ---
    try:
        if not isinstance(fire_df, pd.DataFrame):
            raise TypeError(f"fire_df must be a pandas DataFrame, got {type(fire_df).__name__}.")
        if not isinstance(dispatch_df, pd.DataFrame):
            raise TypeError(f"dispatch_df must be a pandas DataFrame, got {type(dispatch_df).__name__}.")
        if not isinstance(zone_column, str):
            raise TypeError(f"zone_column must be a string, got {type(zone_column).__name__}.")
        if not isinstance(fire_id_column, str):
            raise TypeError(f"fire_id_column must be a string, got {type(fire_id_column).__name__}.")
        if zone_column not in dispatch_df.columns:
            raise ValueError(f"zone_column {zone_column} not in dispatch_df.columns")
        if fire_id_column not in fire_df.columns:
            raise ValueError(f"fire_id_column {fire_id_column} not in fire_df.columns")
    except Exception:
        logger.exception("Invalid input to map_zones_from_dispatch_to_fires.")
        raise

    # --- Map zones ---
    logger.info("Mapping zones from dispatch to fires.")
    fire_df = fire_df.copy()
    dispatch_df = dispatch_df.copy()
    dispach_zone_series = dispatch_df.groupby(fire_id_column)[zone_column].first()
    maped_zones_series = fire_df[fire_id_column].map(dispach_zone_series)

    if zone_column in fire_df.columns:
        fire_df = fire_df.drop(columns=[zone_column], axis=1)

    fire_df.insert(3, zone_column, maped_zones_series)

    logger.info("Successfully mapped zones from dispatch to fires.")
    return fire_df


def filter_target_zones(
    fire_df: pd.DataFrame,
    target_zones: dict = {
        "UNIDAD BASICA": "BASICA",
        "ZONA INTERFAZ": "INTERFAZ",
        "CONFLICTO": "CONFLICTO",
        "VALDIVIA": "VALDIVIA",
        "NOCTURNO": "NOCTURNO",
    },
    target_zones_alt: str = "OTRO",
    target_zone_column: str = "target_zone",
) -> pd.DataFrame:
    """ "
    Filter a DataFrame to only fires with given target zone.

    Parameters:
        fire_df (pd.DataFrame): The DataFrame to filter.
        target_zones (dict): The dictionary of target zones to filter on.
        target_zones_alt (str): The alternative target zone to filter on.
        target_zone_column (str): The name of the column to filter on.

    Returns:
        fire_df (pd.DataFrame): The filtered DataFrame.
    """

    # --- Call args (debug) ---
    logger.debug("called with fire_df.shape=%s, target_zones=%s, target_zones_alt=%s, target_zone_column=%s", fire_df.shape, target_zones, target_zones_alt, target_zone_column)

    # --- Validate input ---
    try:
        if not isinstance(fire_df, pd.DataFrame):
            raise TypeError(f"fire_df must be a pandas DataFrame, got {type(fire_df).__name__}.")
        if not isinstance(target_zones, dict):
            raise TypeError(f"target_zones must be a dictionary, got {type(target_zones).__name__}.")
        if not isinstance(target_zones_alt, str):
            raise TypeError(f"target_zones_alt must be a string, got {type(target_zones_alt).__name__}.")
        if not isinstance(target_zone_column, str):
            raise TypeError(f"target_zone_column must be a string, got {type(target_zone_column).__name__}.")
        if target_zone_column not in fire_df.columns:
            raise ValueError(f"target_zone_column {target_zone_column} not in fire_df.columns")
    except Exception:
        logger.exception("Invalid input to filter_target_zones.")
        raise

    # --- Filter target zones ---
    logger.info("Filtering target zones.")
    fire_df = fire_df.copy()

    # Map target zones
    fire_df.loc[:, target_zone_column] = fire_df[target_zone_column].map(lambda x: target_zones[x] if x in target_zones else target_zones_alt)
    # Filter out zones that are not in the target zones
    fire_df = fire_df[fire_df[target_zone_column] != target_zones_alt]

    logger.info("Successfully filtered target zones.")
    return fire_df


def normalize_and_validate_surface_area_data(fire_df: pd.DataFrame) -> pd.DataFrame:
    """"
    Normalize and validate surface area data in a DataFrame.
    
    Parameters:
        fire_df (pd.DataFrame): The DataFrame to normalize and validate.
        
    Returns:
        pd.DataFrame: The validated DataFrame.
    """

    # --- Call args (debug) ---
    logger.debug("called with fire_df.shape=%s", fire_df.shape)

    # --- Validate input ---
    try:
        if not isinstance(fire_df, pd.DataFrame):
            raise TypeError(f"fire_df must be a pandas DataFrame, got {type(fire_df).__name__}.")
    except Exception:
        logger.exception("Invalid input to normalize_and_validate_surface_area_data.")
        raise

    # --- Normalize and validate surface area data ---
    logger.info("Normalizing surface area data.")
    fire_df = fire_df.copy()
    
    # Normalize
    if fire_df['initial_surface_ha'].dtype == 'object':
        fire_df.loc[:, 'initial_surface_ha'] = fire_df['initial_surface_ha'].str.replace('.', '').str.replace(',', '.').astype(float)
    if fire_df['final_surface_ha'].dtype == 'object':
        fire_df.loc[:, 'final_surface_ha'] = fire_df['final_surface_ha'].str.replace('m2', '').str.replace('.', '').str.replace(',', '.').astype(float)

    # Validate
    if (fire_df['initial_surface_ha'].dropna() < 0).any():
        raise ValueError("Initial surface area has negative values.")
    if (fire_df['final_surface_ha'].dropna() < 0).any():
        raise ValueError("Final surface area has negative values.")
    
    surface_area_diff = fire_df[['initial_surface_ha', 'final_surface_ha']].dropna()
    if (surface_area_diff['final_surface_ha'] < surface_area_diff['initial_surface_ha']).any():
        raise ValueError("Final surface area is greater than initial surface area.")
    
    logger.info("Successfully normalized and validated surface area data.")
    return fire_df


def reorder_hr_columns(dispatch_df: pd.DataFrame) -> pd.DataFrame:
    """"
    Reorder HR columns in a DataFrame.
    
    Parameters:
        dispatch_df (pd.DataFrame): The DataFrame to reorder.
        
    Returns:
        pd.DataFrame: The reordered DataFrame.
    """
    # --- Call args (debug) ---
    logger.debug("called with dispatch_df.shape=%s", dispatch_df.shape)
    
    # --- Validate input ---
    try:
        if not isinstance(dispatch_df, pd.DataFrame):
            raise TypeError(f"dispatch_df must be a pandas DataFrame, got {type(dispatch_df).__name__}.")
    except Exception:
        logger.exception("Invalid input to reorder_hr_columns.")
        raise
    
    # --- Reorder HR columns ---
    logger.info("Reordering HR columns.")
    dispatch_df = dispatch_df.copy()
    
    cols = [c for c in dispatch_df.columns if c.startswith("hr_")]

    for c in cols:
        dispatch_df.loc[:,c] = pd.to_datetime(dispatch_df[c].astype(str).str.strip(), format="%Y-%m-%d %H:%M", errors="coerce")

    #dispatch_df = dispatch_df.dropna(subset=cols)
    
    reordered_df = dispatch_df.sort_values(by=cols, ascending=True)
    
    logger.info("Successfully reordered HR columns.")
    return reordered_df


def merge_dataframes(fire_df: pd.DataFrame, dispatch_df: pd.DataFrame, fire_df_columns: list[str] = ['fire_id', 'start_datetime', 'arrival_datetime_inc', 'control_datetime', 'season'], dispatch_df_columns: list[str] = None, on: str = "fire_id") -> pd.DataFrame:
    """"
    Merge two DataFrames based on a common id column.
    
    Parameters:
        fire_df (pd.DataFrame): The first DataFrame to merge.
        dispatch_df (pd.DataFrame): The second DataFrame to merge.
        fire_df_columns (list[str]): The list of columns to keep from fire_df.
        dispatch_df_columns (list[str]): The list of columns to keep from dispatch_df.
        on (str): The column to merge on.
        
    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    
    # --- Call args (debug) ---
    logger.debug("called with fire_df.shape=%s, dispatch_df.shape=%s, fire_df_columns=%s, dispatch_df_columns=%s, on=%s", fire_df.shape, dispatch_df.shape, fire_df_columns, dispatch_df_columns, on)
    
    # --- Validate input ---
    try:
        if not isinstance(fire_df, pd.DataFrame):
            raise TypeError(f"fire_df must be a pandas DataFrame, got {type(fire_df).__name__}.")
        if not isinstance(dispatch_df, pd.DataFrame):
            raise TypeError(f"dispatch_df must be a pandas DataFrame, got {type(dispatch_df).__name__}.")
        if not isinstance(fire_df_columns, list):
            raise TypeError(f"fire_df_columns must be a list, got {type(fire_df_columns).__name__}.")
        if not isinstance(dispatch_df_columns, list):
            raise TypeError(f"dispatch_df_columns must be a list, got {type(dispatch_df_columns).__name__}.")
        if not isinstance(on, str):
            raise TypeError(f"on must be a string, got {type(on).__name__}.")
        if on not in fire_df.columns:
            raise KeyError(f"Column '{on}' not found in fire_df.")
        if on not in dispatch_df.columns:
            raise KeyError(f"Column '{on}' not found in dispatch_df.")
    except Exception:
        logger.exception("Invalid input to merge_dataframes.")
        raise
    
    # --- Merge ---
    logger.info("Merging dataframes.")
    fire_df = fire_df.copy()
    dispatch_df = dispatch_df.copy()
    
    right_df = fire_df[fire_df_columns]
    
    for col in fire_df_columns:
        if col == on:
            continue
        if col in dispatch_df.columns:
            dispatch_df = dispatch_df.drop(columns=[col])
        if col.endswith('_datetime'):
            right_df.loc[:, col] = pd.to_datetime(right_df[col].astype(str).str.strip(), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    
                
    merged_df = pd.merge(left=dispatch_df, right=right_df, on=on, how='left')

    logger.info("Successfully merged dataframes.")
    return merged_df


def detect_and_fix_time_anomalies(
    fire_df: pd.DataFrame,
    dispatch_df: pd.DataFrame,
    *,
    fire_id_col: str = "fire_id",
    arrival_col: str = "arrival_datetime_inc",
    start_col: str = "start_datetime",          # adjust if your pipeline uses start_datetime_inc
    hr_arribo_col: str = "hr_arribo",
    hr_prefix: str = "hr_",
    fix_dst: bool = True,
    return_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Detect DST 60-min anomalies and order errors, optionally fix DST by shifting all hr_* by +60 min
    for affected fires. Returns (dispatch_df_fixed, report_df) if return_report else just dispatch_df_fixed.
    """
    logger.info("detect_and_fix_time_anomalies: start")

    # ---- validation
    if not isinstance(fire_df, pd.DataFrame) or not isinstance(dispatch_df, pd.DataFrame):
        raise TypeError("fire_df and dispatch_df must be pandas DataFrames.")
    for name, df, cols in (
        ("fire_df", fire_df, {fire_id_col, arrival_col, start_col}),
        ("dispatch_df", dispatch_df, {fire_id_col, hr_arribo_col}),
    ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} missing required columns: {missing}")

    hr_cols = [c for c in dispatch_df.columns if c.startswith(hr_prefix)]
    if not hr_cols:
        logger.warning("detect_and_fix_time_anomalies: no '%s*' columns found; continuing without shifting.", hr_prefix)

    # ---- normalize datetimes
    fd = fire_df[[fire_id_col, arrival_col, start_col]].copy()
    fd[arrival_col] = pd.to_datetime(fd[arrival_col], errors="coerce")
    fd[start_col] = pd.to_datetime(fd[start_col], errors="coerce")

    dd = dispatch_df.copy()
    dd[hr_arribo_col] = pd.to_datetime(dd[hr_arribo_col], errors="coerce")
    for c in hr_cols:
        dd[c] = pd.to_datetime(dd[c], errors="coerce")

    # ---- collapse to earliest arrival per fire on the dispatch side
    arr_min = dd.groupby(fire_id_col, dropna=True)[hr_arribo_col].min()

    rep = fd.copy()
    rep[hr_arribo_col] = rep[fire_id_col].map(arr_min)
    rep["time_diff_min_before"] = (rep[arrival_col] - rep[hr_arribo_col]).dt.total_seconds() / 60
    rep["dst_flag_before"] = rep["time_diff_min_before"].round().eq(60)

    # ---- apply DST shift if requested
    affected_ids = rep.loc[rep["dst_flag_before"], fire_id_col].dropna().unique()
    if fix_dst and affected_ids.size and hr_cols:
        idx = dd[fire_id_col].isin(affected_ids)
        dd.loc[idx, hr_cols] = dd.loc[idx, hr_cols] + pd.Timedelta(minutes=60)

        # recompute report against adjusted arrivals
        arr_min2 = dd.groupby(fire_id_col, dropna=True)[hr_arribo_col].min()
        rep[hr_arribo_col] = rep[fire_id_col].map(arr_min2)
        rep["time_diff_min_after"] = (rep[arrival_col] - rep[hr_arribo_col]).dt.total_seconds() / 60
        rep["dst_fixed"] = rep["dst_flag_before"] & rep["time_diff_min_after"].round().eq(0)
    else:
        rep["time_diff_min_after"] = rep["time_diff_min_before"]
        rep["dst_fixed"] = False

    # ---- order error after any DST fix
    rep["order_error"] = rep[start_col] > rep[hr_arribo_col]

    # ---- summary logs
    n_dst = int(rep["dst_flag_before"].sum())
    n_fixed = int(rep["dst_fixed"].sum())
    n_order = int(rep["order_error"].sum())
    logger.info(
        "detect_and_fix_time_anomalies: dst_flag_before=%d, dst_fixed=%d, order_error=%d, rows=%d",
        n_dst, n_fixed, n_order, len(rep),
    )

    if return_report:
        cols = [fire_id_col, arrival_col, start_col, hr_arribo_col,
                "time_diff_min_before", "time_diff_min_after", "dst_flag_before", "dst_fixed", "order_error"]
        return dd, rep[cols]
    return dd

def merge_and_analyze(df, dfd):

    left = df[['fire_id', 'arrival_datetime_inc']].copy()
    right = dfd[['fire_id', 'hr_arribo']].copy()

    # cast to datetime
    left['arrival_datetime_inc'] = pd.to_datetime(left['arrival_datetime_inc'])
    right['hr_arribo'] = pd.to_datetime(right['hr_arribo'])
    right = right.groupby('fire_id').min().reset_index()
    merged = pd.merge(left, right, on='fire_id', how='left')

    # error
    merged['time_diff'] = (merged['arrival_datetime_inc'] - merged['hr_arribo']).dt.total_seconds() / 60

    print(f"Number of records with positive time difference: {merged[merged['time_diff'] > 0].shape[0]} out of {merged.shape[0]}")
    print(f"Number of records with negative time difference: {merged[merged['time_diff'] < 0].shape[0]} out of {merged.shape[0]}")
    print(f"Number of records with zero time difference: {merged[merged['time_diff'] == 0].shape[0]} out of {merged.shape[0]}")


def preprocessing_pipeline(fire_data_paths: list[str], dispatch_data_paths: list[str], output_dir: str = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load datasets
    fire_dfs = load_datasets(fire_data_paths)
    dispatch_dfs = load_datasets(dispatch_data_paths)
    
    # Check compatibility of dataframes
    check_dataset_compatibility(fire_dfs, check_unique_rows=True)
    check_dataset_compatibility(dispatch_dfs)
    
    # Concatenate and rename dataframes
    fire_df = pd.concat(fire_dfs, ignore_index=True)
    fire_df_name = concatnate_name_from_paths(fire_data_paths)
    fire_df.attrs["name"] = fire_df_name
    
    # Convert and add date time columns, and standardize fire IDs
    fire_df = convert_and_add_date_time_columns(fire_df)
    fire_df = standardize_fire_reports_IDs(fire_df)
    
    # Standardize dispatch IDs and filter, and standardize response teams
    dispatch_dfs = [standardize_dispatch_reports_ID_and_fliter(dispatch_df) for dispatch_df in dispatch_dfs]
    dfd1, dfd2 = standardize_response_teams(dispatch_dfs[0], dispatch_dfs[1])
    
    
    # Concatenate and rename dataframes
    dispatch_df = pd.concat([dfd1, dfd2], ignore_index=True)
    dispatch_df_name = concatnate_name_from_paths(dispatch_data_paths)
    dispatch_df.attrs["name"] = dispatch_df_name
    
    #fire_summary_by_season(fire_df, dfd1)
    
    
    # Match fires and dispatches, and select and rename variables
    fire_df, dispatch_df = match_fires_and_dispatches(fire_df, dispatch_df)
    fire_df, dispatch_df = selecting_and_renaming_fire_variables(fire_df, dispatch_df)
    
    # Map zones, filter target zones and normalize surface area data
    fire_df = map_zones_from_dispatch_to_fires(fire_df, dispatch_df)
    fire_df = filter_target_zones(fire_df)
    fire_df = normalize_and_validate_surface_area_data(fire_df)
    
    
    # Reorder HR columns and merge
    dispatch_df = reorder_hr_columns(dispatch_df)
    dispatch_df = merge_dataframes(fire_df, dispatch_df)
    
    
    # Daylight saving time error
    dispatch_df = detect_and_fix_time_anomalies(fire_df, dispatch_df)
    
    
    merge_and_analyze(fire_df, dispatch_df)
    
    if output_dir is not None:
        # Check directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save dataframes to CSV files.
        fire_df.to_csv(f"{output_dir}/fire_data_2014-2025.csv", sep=";", index=False)
        dispatch_df.to_csv(f"{output_dir}/dispatch_data_2014-2025.csv", sep=";", index=False)
        
        logger.info("Successfully saved preprocessed data to %s.", output_dir)
    

    return fire_df, dispatch_df
