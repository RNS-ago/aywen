import math
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from aywen.fire_features import  COVARIATES, COVARIATES_CATEGORICAL, CODE_KITRAL
from aywen.utils import replace_items
from typing import Dict, Tuple, Hashable, Optional
import pandas as pd

logger = logging.getLogger("aywen_logger")


# Column definitions
ZONE_ALERT_COL = 'zone_alert'

ZONE_FACTOR_COLS = ['zone_WE', 'zone_NS']
ZONE_FACTOR_LEVELS = {
    'zone_WE': ['Costa', 'Valle', 'Cordillera'],
    'zone_NS': ['1', '2', '3', '4', '5']
}

COVARIATES = replace_items(COVARIATES, {
    "initial_fuel": "initial_fuel_reduced"
})
COVARIATES_CATEGORICAL = replace_items(COVARIATES_CATEGORICAL, {
    "initial_fuel": "initial_fuel_reduced"
})


def filter_rows_by_range(df, thresholds):
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

def postprocess_zone_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess zone factors (zone_WE and zone_NS) by handling missing values
    and converting to categorical variables.
    """
    df = df.copy()
    
    # Handle missing values in zone factors
    if df[ZONE_FACTOR_COLS].isnull().any().any():
        initial_rows = len(df)
        df = df.dropna(subset=ZONE_FACTOR_COLS, how='any')
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing zone factors")
    
    if df.empty:
        raise ValueError("No rows remaining after filtering zone factors")
    
    # Clean and validate zone_NS (ensure it's clean integer-like strings)
    df['zone_NS'] = pd.to_numeric(df['zone_NS'], errors='coerce').astype('Int64').astype(str)
    
    # Validate factor levels
    for factor_col in ZONE_FACTOR_COLS:
        expected_levels = ZONE_FACTOR_LEVELS[factor_col]
        actual_levels = set(df[factor_col].astype(str).unique())
        unexpected_levels = actual_levels - set(expected_levels)
        
        if unexpected_levels:
            raise ValueError(f"{factor_col} contains unexpected levels: {sorted(unexpected_levels)}")
        
        # Convert to categorical with explicit ordering
        df[factor_col] = pd.Categorical(df[factor_col], categories=expected_levels, ordered=True)
    
    logger.info("Successfully preprocessed zone factors")
    return df

def filter_low_observation_zones(df, min_observations):
    """Filter out zones with few observations."""
    df = df.copy()
    initial_rows = len(df)
    
    # Get zones that have more than min_observations
    zone_counts = df[ZONE_ALERT_COL].value_counts()
    valid_zones = zone_counts[zone_counts > min_observations].index
    
    # Filter dataframe to only include valid zones
    df = df[df[ZONE_ALERT_COL].isin(valid_zones)]
    
    logger.info(f"Filtered zones with <={min_observations} observations: "
                f"{initial_rows - len(df)} rows removed")
    return df

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
        fuel_col: str = 'initial_fuel_reduced',
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
    df_chk = out.dropna(subset=[fuel_col])

    counts = (df_chk
        .groupby([factor1, factor2, fuel_col, 'split3'], observed=True)
        .size()
        .unstack('split3', fill_value=0)
    )

    # ensure train column exists even if there are no train rows at all
    if 'train' not in counts.columns:
        counts['train'] = 0

    # rows present somewhere (they all are) but missing in train
    missing = counts[counts['train'] == 0]

    # assert missing.empty, (
    #     "Some initial_fuel_reduced levels are missing from the train split:\n"
    #     + missing.reset_index()[[factor1, factor2, fuel_col]].to_string(index=False)
    # )

    logger.info("Added %s and %s.", split2_col, split3_col)

    return out


# -------- reduced fuel  -------

def _old_reduce_fuel_types(s: pd.Series, min_fuel_samples) -> pd.Series:
    """Collapse rare levels in s to 'other'. Accepts int or fraction."""
    if min_fuel_samples < 1:
        n = s.size
        min_fuel_samples = max(1, math.ceil(min_fuel_samples * n))
        logger.debug("min_fuel_samples as fraction -> %d samples", min_fuel_samples)

    counts = s.value_counts()
    rare = counts[counts < min_fuel_samples].index
    logger.debug("Number of rare fuel types: %s", len(rare))
    s.cat.rename_categories(lambda x: x.replace(rare, 'other'), inplace=True)
    return s.replace(rare, 'other')


def _reduce_fuel_types(s: pd.Series, min_fuel_samples, other_label: str = "other") -> pd.Series:
    """
    Collapse rare levels in s to `other_label`. `min_fuel_samples` accepts an int or a fraction in (0,1).
    Returns a categorical Series with unused categories removed.
    """
    s = s.copy()

    # Ensure categorical dtype (so we keep categories after recoding)
    if not isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype("category")

    # Convert fraction -> absolute threshold
    if isinstance(min_fuel_samples, (float, int)) and min_fuel_samples < 1:
        n = s.size
        min_fuel_samples = max(1, math.ceil(min_fuel_samples * n))

    # Count per level (NaNs excluded by default; change dropna=False if desired)
    counts = s.value_counts(dropna=True)
    rare = counts[counts < min_fuel_samples].index

    # Optionally avoid collapsing the target label itself if present
    if len(rare) == 0:
        return s

    # 1) Make sure `other_label` is a valid category
    if other_label not in s.cat.categories:
        s = s.cat.add_categories([other_label])

    # 2) Assign rare levels to "other" (no .replace to avoid FutureWarning)
    mask = s.isin(rare)
    if mask.any():
        s.loc[mask] = other_label

    # 3) Drop categories that are no longer used
    s = s.cat.remove_unused_categories()

    return s



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

    # out['tmp'] = (out[fuel_col]
    #    .astype(str)
    #    .str.replace('-', '', regex=False)
    #    .str.replace(' ', '_', regex=False)
    #    .str.replace(',', '_', regex=False)
    #    .str.replace('>', 'gt_', regex=False)
    #    .str.replace('<', 'lt_', regex=False)
    #    .str.replace('=', 'eq_', regex=False)
    #    .str.replace('.', '', regex=False)
    #    .str.lower()
    # )

    # groupwise reduction (aligned back to original index)
    out[fuel_reduced_col] = out[fuel_col].groupby([out[factor1], out[factor2]], observed=True).transform(lambda g: _reduce_fuel_types(g, min_fuel_samples=min_fuel_samples))
    #out = out.drop(columns=['tmp'])
    out[fuel_reduced_col] = pd.Categorical(out[fuel_reduced_col], categories=out[fuel_reduced_col].unique())

    kept = out[fuel_reduced_col].groupby([out[factor1], out[factor2]], observed=True).nunique(dropna=True)
    original = df[fuel_col].groupby([df[factor1], df[factor2]], observed=True).nunique(dropna=True)
    logger.debug("Fuel categories reduced per group: \n%s", pd.DataFrame({'original': original, 'kept': kept}))

    logger.info("Added %s.", fuel_reduced_col)

    return out
    

def learn_groupwise_mapping(
    df: pd.DataFrame,
    factor1: str = "zone_WE",
    factor2: str = "zone_NS",
    categories: Optional[list] = None,            # infer if None
    learn_source_col: str = "initial_fuel",
    learn_target_col: str = "initial_fuel_reduced",
    other_label: str = "other",                   # match your pipeline naming
    strict: bool = True,                          # raise on conflicts
) -> Dict[Tuple[Hashable, Hashable], Dict[Hashable, Hashable]]:
    """
    Learn mapping from `learn_source_col` → `learn_target_col` within each (factor1, factor2).

    Returns:
        Dict[(factor1, factor2) -> Dict[source_category -> target_category]]
    """

    logger.important("Learning groupwise mapping from %s to %s within (%s, %s)",
        learn_source_col, learn_target_col, factor1, factor2
    )

    # --- Validate inputs
    required = {factor1, factor2, learn_source_col, learn_target_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in df: {sorted(missing)}")

    # --- Decide the universe of source categories
    if categories is None:
        categories = (
            df[learn_source_col].dropna().unique().tolist()
        )

    # --- Pre-create mapping for each observed (factor1, factor2)
    groups = df[[factor1, factor2]].drop_duplicates()
    mapping: Dict[Tuple[Hashable, Hashable], Dict[Hashable, Hashable]] = {
        (row[factor1], row[factor2]): {cat: other_label for cat in categories}
        for _, row in groups.iterrows()
    }

    # --- Fill with learned pairs (single pass)
    for (f1, f2), g in df.groupby([factor1, factor2], observed=True, dropna=False):
        # Keep only valid pairs
        sub = g[[learn_source_col, learn_target_col]].dropna()

        # Remove duplicate rows
        sub = sub.drop_duplicates()

        if strict:
            # Detect conflicts: same source → multiple targets in this group
            conflict_counts = sub.groupby(learn_source_col, observed=True)[learn_target_col].nunique()
            conflicts = conflict_counts[conflict_counts > 1]
            if not conflicts.empty:
                masked = sub[sub[learn_source_col].isin(conflicts.index)]
                raise ValueError(
                    f"Conflicting mappings in group {(f1, f2)} for "
                    f"{learn_source_col}: {masked.sort_values(learn_source_col).to_dict(orient='list')}"
                )

        # Build pairs (if multiple rows per source remain, take first)
        pairs = (
            sub.groupby(learn_source_col, observed=True)[learn_target_col]
               .first()
               .to_dict()
        )

        # Update the group mapping
        mapping[(f1, f2)].update(pairs)

    return mapping


def add_groupwise_mapping(
    df: pd.DataFrame,
    *,
    factor1: str = "zone_WE",
    factor2: str = "zone_NS",
    mapping: Dict[Tuple[Hashable, Hashable], Dict[Hashable, Hashable]],
    source_col: str,
    target_col: str,
    unknown_label: str = "Unknown",
    fill_unmapped: bool = True,   # if False, leave unmapped as NaN
    cast_target_to_category: bool = False,
) -> pd.DataFrame:
    """
    Apply a pre-learned mapping from `source_col` → `target_col` within each (factor1, factor2).
    """

    # --- Validate
    if mapping is None:
        raise ValueError("`mapping` must be provided (dict[(f1,f2) -> dict[source->target]]).")
    for col in (factor1, factor2, source_col):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col!r}")

    # --- Build a tidy mapping table for a vectorized merge
    # rows: factor1, factor2, source_col, target_col
    rows = []
    for (f1, f2), d in mapping.items():
        # guard: empty dicts are ok; they just won't map anything
        for src, tgt in d.items():
            rows.append({factor1: f1, factor2: f2, source_col: src, target_col: tgt})
    map_df = pd.DataFrame(rows, columns=[factor1, factor2, source_col, target_col])

    out = df.copy()

    if map_df.empty:
        # no learned pairs: either keep NaN or fill with unknown
        out[target_col] = unknown_label if fill_unmapped else pd.NA
    else:
        # --- Left join to attach mapped targets
        out = out.merge(
            map_df,
            how="left",
            on=[factor1, factor2, source_col],
            suffixes=("", "_mapped"),
        )
        # after merge, target_col holds the mapped value

        if fill_unmapped:
            out[target_col] = out[target_col].fillna(unknown_label)

        if cast_target_to_category:
            # categories from the mapping + maybe the unknown label
            cats = pd.Series(map_df[target_col].dropna().unique()).tolist()
            if fill_unmapped and unknown_label not in cats:
                cats.append(unknown_label)
            out[target_col] = out[target_col].astype(pd.CategoricalDtype(categories=cats))

    # Optional, compact logging
    try:
        n_pairs = sum(len(d) for d in mapping.values())
        logger.info(
            "Added %s using %d groups and %d pairs.",
            target_col, len(mapping), n_pairs
        )
    except Exception:
        logger.info("Added %s.", target_col)

    return out



# ------- postprocessing pipeline -------

def postprocessing_pipeline(
        df, 
        thresholds=None, 
        zone_col="zone_alert",
        fuel_col: str = "initial_fuel",
        fuel_reduced_col: str = "initial_fuel_reduced"
    ) -> pd.DataFrame:


    if thresholds is None:
        thresholds = {
            "dt_minutes": {"lower": 10, "upper": 30},
            "initial_radius_m": {"lower": 2, "upper": 100}
        }

    logger.important("Starting postprocessing pipeline with df.shape=%s", df.shape)

    out = df.copy()
    out = filter_rows_by_range(out, thresholds)
    out = postprocess_zone_factors(out)
    out = filter_low_observation_zones(out, min_observations=5)
    out = drop_missing_rows(out)
    out = add_fuel_reduced(out, fuel_col=fuel_col, fuel_reduced_col=fuel_reduced_col)
    out = add_train_test_split_to_df(out, zone_col=zone_col, fuel_col=fuel_reduced_col)

    mapping = learn_groupwise_mapping(out, learn_source_col=fuel_col, learn_target_col=fuel_reduced_col, categories=list(CODE_KITRAL.values()))

    return out, mapping
