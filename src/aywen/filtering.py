import pandas as pd
import numpy as np
import math
import logging
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional, Callable
import os

logger = logging.getLogger("wildfire_processor")

# Column definitions
FIRE_ID_COL = 'fire_id'
ZONE_ALERT_COL = 'zone_alert'

GEOSPATIAL_COLS = ['longitude', 'latitude']
ZONE_FACTOR_COLS = ['zone_WE', 'zone_NS']

COVARIATES_NUMERIC_COLS = [
    'temperature', 
    'wind_speed_kmh', 
    'relative_humidity', 
    'ffdi',
    'slope_degrees',
    'sin_hour',
    'cos_hour',
    'slope_wind_synergy'
]

COVARIATES_CATEGORICAL_COLS = [
    'high_season', 
    'day_night',
    'initial_fuel',
    'weather_index_full'
]

TARGET_COLS = [
    'dt_minutes', 
    'initial_radius_m', 
    'initial_surface_ha',
    'propagation_speed_mm'
]

# Expected factor levels
ZONE_FACTOR_LEVELS = {
    'zone_WE': ['Costa', 'Valle', 'Cordillera'],
    'zone_NS': ['1', '2', '3', '4', '5']
}

def load_wildfire_data(file_path: str) -> pd.DataFrame:
    """Load wildfire data from CSV file."""
    try:
        usecols = ([FIRE_ID_COL] + GEOSPATIAL_COLS + [ZONE_ALERT_COL] +
                  ZONE_FACTOR_COLS + COVARIATES_NUMERIC_COLS + 
                  COVARIATES_CATEGORICAL_COLS + TARGET_COLS)
        
        data = pd.read_csv(file_path, sep=',', usecols=usecols)
        logger.info(f"Loaded data: {len(data)} rows, {len(data.columns)} columns")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def preprocess_zone_factors(df: pd.DataFrame) -> pd.DataFrame:
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

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing values."""
    initial_rows = len(df)
    df = df.dropna(how='any')
    logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")
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

def apply_range_filters(df: pd.DataFrame, 
                       dt_minutes_range: Tuple[float, float] = (10, 30),
                       initial_radius_range: Tuple[float, float] = (2, 100)) -> pd.DataFrame:
    """Apply filtering criteria to the dataset."""
    df = df.copy()
    initial_rows = len(df)
    
    # Filter dt_minutes
    df = df[(df['dt_minutes'] >= dt_minutes_range[0]) & 
            (df['dt_minutes'] <= dt_minutes_range[1])]
    logger.info(f"Applied dt_minutes filter [{dt_minutes_range[0]}, {dt_minutes_range[1]}]: "
               f"{len(df)} rows remaining")
    
    # Filter initial_radius_m
    df = df[(df['initial_radius_m'] >= initial_radius_range[0]) & 
            (df['initial_radius_m'] <= initial_radius_range[1])]
    logger.info(f"Applied initial_radius_m filter [{initial_radius_range[0]}, {initial_radius_range[1]}]: "
               f"{len(df)} rows remaining")
    
    logger.info(f"Total rows filtered: {initial_rows - len(df)}")
    return df

def sanitize_fuel_types(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize initial_fuel column by cleaning text values."""
    df = df.copy()
    df['initial_fuel_sanitized'] = (df['initial_fuel']
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
    
    logger.info("Sanitized fuel types")
    return df

def reduce_fuel_types(series: pd.Series, min_fuel_samples: int) -> pd.Series:
    """Collapse rare fuel type levels to 'other'."""
    if min_fuel_samples < 1:
        n = series.size
        min_fuel_samples = max(1, math.ceil(min_fuel_samples * n))
        logger.info(f"Minimum fuel samples: {min_fuel_samples}")
    
    counts = series.value_counts()
    rare = counts[counts < min_fuel_samples].index
    return series.replace(rare, 'other')

def create_reduced_fuel_types(df: pd.DataFrame, min_fuel_samples: int = 20) -> pd.DataFrame:
    """Create reduced fuel types by zone, collapsing rare categories."""
    df = df.copy()
    
    # Group-wise reduction aligned back to original index
    df['initial_fuel_reduced'] = df['initial_fuel_sanitized'].groupby(
        [df['zone_WE'], df['zone_NS']], observed=True
    ).transform(lambda g: reduce_fuel_types(g, min_fuel_samples))
    
    logger.info(f"Created reduced fuel types with min_samples={min_fuel_samples}")
    return df

def split_data_by_zone(df: pd.DataFrame, 
                      test_frac: float = 0.20, 
                      val_frac: float = 0.20, 
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets with stratification by zone.
    Splits are performed independently within each zone.
    """
    parts = []
    
    for zone, df_zone in df.groupby(ZONE_ALERT_COL, observed=False):
        # Test split within this zone
        df_temp, df_test = train_test_split(
            df_zone, test_size=test_frac, random_state=random_state, shuffle=True
        )
        
        # Validation split within the remaining data
        val_size_rel = val_frac / (1.0 - test_frac)
        df_train, df_val = train_test_split(
            df_temp, test_size=val_size_rel, random_state=random_state, shuffle=True
        )
        
        # Add dataset labels
        df_train = df_train.copy()
        df_train["dataset"] = "train"
        df_val = df_val.copy()
        df_val["dataset"] = "val"
        df_test = df_test.copy()
        df_test["dataset"] = "test"
        
        parts.append((df_train, df_val, df_test))
    
    # Concatenate all zones back together
    df_train = pd.concat([p[0] for p in parts]).sort_index()
    df_val = pd.concat([p[1] for p in parts]).sort_index()
    df_test = pd.concat([p[2] for p in parts]).sort_index()
    
    logger.info(f"Data split - Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    
    # Validate splits are disjoint
    assert set(df_train.index).isdisjoint(df_val.index)
    assert set(df_train.index).isdisjoint(df_test.index)
    assert set(df_val.index).isdisjoint(df_test.index)
    
    return df_train, df_val, df_test

def add_split_flags(df: pd.DataFrame, 
                   df_train: pd.DataFrame, 
                   df_val: pd.DataFrame, 
                   df_test: pd.DataFrame) -> pd.DataFrame:
    """Add split indicator columns to the main dataframe."""
    df = df.copy()
    
    # Three-way split
    df["split3"] = "train"
    df.loc[df.index.isin(df_val.index), "split3"] = "valid"
    df.loc[df.index.isin(df_test.index), "split3"] = "test"
    
    # Two-way split (train+valid vs test)
    df['split2'] = "train+valid"
    df.loc[df.index.isin(df_test.index), "split2"] = "test"
    
    return df

def validate_fuel_types_in_splits(df: pd.DataFrame) -> None:
    """Validate that all fuel types present in validation/test are also in train."""
    df_check = df.dropna(subset=['initial_fuel_reduced'])
    
    counts = (df_check
        .groupby(['zone_WE', 'zone_NS', 'initial_fuel_reduced', 'split3'], observed=True)
        .size()
        .unstack('split3', fill_value=0)
    )
    
    # Ensure train column exists
    if 'train' not in counts.columns:
        counts['train'] = 0
    
    # Find categories missing from train split
    missing = counts[counts['train'] == 0]
    
    if not missing.empty:
        missing_info = missing.reset_index()[['zone_WE', 'zone_NS', 'initial_fuel_reduced']]
        raise ValueError(
            "Some fuel types are missing from the train split:\n"
            + missing_info.to_string(index=False)
        )
    
    logger.info("Fuel type validation passed - all categories present in train split")

def get_fuel_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics of fuel types per zone in training data."""
    train_df = df[df['split3'] == 'train'].copy()
    
    counts_train = (train_df.dropna(subset=['initial_fuel_reduced'])
                   .groupby(['zone_WE', 'zone_NS', 'initial_fuel_reduced'], observed=True)
                   .size()
                   .rename('n')
                   .reset_index())
    
    # Minimum among present categories in each zone
    grp = counts_train.groupby(['zone_WE', 'zone_NS'], observed=True)['n']
    min_present_train = grp.min().rename('min_n_present').to_frame()
    
    idx = grp.idxmin()
    which_min_present_train = (counts_train.loc[idx]
        .set_index(['zone_WE', 'zone_NS'])['initial_fuel_reduced']
        .rename('which_min_present'))
    
    summary = (min_present_train
              .join(which_min_present_train)
              .reset_index())
    summary['split'] = 'train'
    
    return summary

def save_processed_data(df: pd.DataFrame, output_dir: str, filename: str = 'processed_wildfire_data.csv') -> str:
    """Save processed data to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, sep=',', index=False)
    logger.info(f"Saved processed data to {output_path}")
    return output_path

# Pipeline function
def wildfire_processing_pipeline(
    file_path: str,
    output_dir: str = "out",
    dt_minutes_range: Tuple[float, float] = (10, 30),
    initial_radius_range: Tuple[float, float] = (2, 100),
    min_fuel_samples: int = 20,
    min_zone_observations: int = 5,
    test_frac: float = 0.20,
    val_frac: float = 0.20,
    random_state: int = 42,
    save_data: bool = True,
    pipeline_steps: Optional[List[Callable]] = None
) -> pd.DataFrame:
    """
    Complete wildfire data processing pipeline using functional approach.
    
    Args:
        file_path: Path to input CSV file
        output_dir: Directory to save processed data
        dt_minutes_range: Range filter for dt_minutes
        initial_radius_range: Range filter for initial_radius_m
        min_fuel_samples: Minimum samples to keep fuel type (else 'other')
        min_zone_observations: Minimum observations per zone to keep
        test_frac: Fraction for test split
        val_frac: Fraction for validation split
        random_state: Random seed for reproducibility
        save_data: Whether to save the processed data
        pipeline_steps: Custom pipeline steps (if None, uses default)
        
    Returns:
        Processed DataFrame with all transformations applied
    """
    logger.info("Starting wildfire data processing pipeline")
    
    # Define default pipeline steps
    if pipeline_steps is None:
        pipeline_steps = [
            lambda df: load_wildfire_data(file_path) if isinstance(df, str) else df,
            preprocess_zone_factors,
            drop_missing_values,
            lambda df: filter_low_observation_zones(df, min_zone_observations),
            lambda df: apply_range_filters(df, dt_minutes_range, initial_radius_range),
            sanitize_fuel_types,
            lambda df: create_reduced_fuel_types(df, min_fuel_samples),
        ]
    
    # Execute pipeline steps
    df = file_path  # Start with file_path, first step will load it
    for i, step in enumerate(pipeline_steps):
        try:
            df = step(df)
            logger.info(f"Completed pipeline step {i+1}/{len(pipeline_steps)}: {step.__name__}")
        except Exception as e:
            logger.error(f"Error in pipeline step {i+1} ({step.__name__}): {e}")
            raise
    
    # Data splitting (separate from main pipeline for clarity)
    df_train, df_val, df_test = split_data_by_zone(df, test_frac, val_frac, random_state)
    
    # Add split flags to main dataframe
    df = add_split_flags(df, df_train, df_val, df_test)
    
    # Validation and summary
    validate_fuel_types_in_splits(df)
    fuel_summary = get_fuel_type_summary(df)
    logger.info("Fuel type summary:\n" + fuel_summary.to_string(index=False))
    
    # Save processed data
    if save_data:
        save_processed_data(df, output_dir)
    
    logger.info(f"Pipeline completed successfully. Final dataset shape: {df.shape}")
    return df
# Convenience functions for different use cases
def quick_process(file_path: str, output_dir: str = "out") -> pd.DataFrame:
    """Quick processing with default parameters."""
    return wildfire_processing_pipeline(file_path, output_dir)

def custom_pipeline(file_path: str, steps: List[Callable], **kwargs) -> pd.DataFrame:
    """Run a custom pipeline with user-defined steps."""
    return wildfire_processing_pipeline(file_path, pipeline_steps=steps, **kwargs)

def process_with_custom_filters(
    file_path: str,
    dt_range: Tuple[float, float],
    radius_range: Tuple[float, float],
    output_dir: str = "out"
) -> pd.DataFrame:
    """Process with custom filter ranges."""
    return wildfire_processing_pipeline(
        file_path, 
        output_dir=output_dir,
        dt_minutes_range=dt_range,
        initial_radius_range=radius_range
    )

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wildfire_processor.py <input_file> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "out"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process the data
    processed_df = quick_process(input_file, output_dir)
    print(f"Processing complete. Final dataset shape: {processed_df.shape}")