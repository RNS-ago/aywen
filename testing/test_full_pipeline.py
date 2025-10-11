# testing/test_pipeline.py
from __future__ import annotations

import os
import warnings
from pathlib import Path
import logging
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# Silence noisy third-party warning from richdem about pkg_resources deprecation
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated",
    category=UserWarning,
    module=r"richdem.*",
)

# --- Pytest collection/skip control ---
try:
    import pytest as _pytest
    from pathlib import Path as _Path
    _dp = globals().get('DATA_PROCESSED')
    _do = globals().get('DATA_ORIGINAL')
    _root = globals().get('DIR_ROOT')
    _paths_ok = True
    for _p in (_dp, _do):
        if _p and not _Path(_p).exists():
            _paths_ok = False
    if _root:
        _paths_ok = _paths_ok and _Path(_root).exists()
    pytestmark = _pytest.mark.skipif(not _paths_ok, reason='Required data paths not available; skipping integration tests.')
except Exception:
    pass


# quiet rasterio
for name in ("rasterio", "rasterio.env", "rasterio._env"):
  lg = logging.getLogger(name)
  lg.setLevel(logging.ERROR)   # or CRITICAL
  lg.propagate = False


# --- Project imports ---
import os
import logging
from pathlib import Path
from aywen.logging_setup import configure_logging
from aywen.preprocessing import preprocessing_pipeline
from aywen.fire_features import feature_engineering_pipeline
from aywen.postprocessing import postprocessing_pipeline
from aywen.fire_features import DEFAULT_COLUMNS_DICT
from aywen.training import train_pipeline,  add_elliptical_propagation_speed_to_df, add_base_model_predictions_to_df
from aywen.testing import assert_df_from_file, assert_predictions_match

# --- Define paths ---
DIR_ROOT =  "G:/Shared drives/OpturionHome/AraucoFire"
DATA_ORIGINAL = DIR_ROOT + "/2_data/original"
DATA_PROCESSED = DIR_ROOT + "/2_data/processed"

FIRE_CSVS = [
    DATA_PROCESSED + "/Incendios_2014-2018.csv",
    DATA_PROCESSED + "/Incendios_2021-2025.csv",
]

DISPATCH_CSVS = [
    DATA_PROCESSED + "/Despachos_2014-2018.csv",
    DATA_PROCESSED + "/Despachos_2021-2025.csv",
]

ZONE_SHAPEFILE_PATH = DATA_ORIGINAL + "/tiles_arauco/Zonas_alerta/Zonas_alerta_.shp"
TOPO_CSV_PATH = DATA_PROCESSED + "/topographic_data.csv"
TIFF_FUEL_PATH = DATA_ORIGINAL + "/tiles_arauco/ModeloCombustible202409/ModeloCombustible202409.tif"

# --- Select columns ---
id = DEFAULT_COLUMNS_DICT["id"]
geospatial = DEFAULT_COLUMNS_DICT["geospatial"]
timestamp = DEFAULT_COLUMNS_DICT["timestamp"]
factors = DEFAULT_COLUMNS_DICT["factors"]
covariates = DEFAULT_COLUMNS_DICT["covariates"]
targets = DEFAULT_COLUMNS_DICT["targets"]
split = DEFAULT_COLUMNS_DICT["split"]
others = DEFAULT_COLUMNS_DICT["others"]
columns = id + geospatial + timestamp + factors + covariates + targets + others

# --- Configure logging ---
configure_logging()
logging.getLogger("aywen_logger").setLevel(logging.WARNING)

# --------- preprocessing pipeline ---------
fire_df, dispatch_df = preprocessing_pipeline(FIRE_CSVS, DISPATCH_CSVS)

# --------- feature engineering pipeline ---------
fire_df_fe = feature_engineering_pipeline(fire_df, zone_shapefile_path=ZONE_SHAPEFILE_PATH, topo_csv_path=TOPO_CSV_PATH, fuel_tiff_path=TIFF_FUEL_PATH)

# --------- postprocessing pipeline ---------
fire_df_post = postprocessing_pipeline(fire_df_fe)

# ---------  training pipeline ---------
factor1 = factors[0]  # "zone_WE"
factor2 = factors[1]  # "zone_NS"
covariates_categorical = DEFAULT_COLUMNS_DICT["covariates_categorical"]
target = "propagation_speed_mm"
pi_covariates = DEFAULT_COLUMNS_DICT["pi_covariates"]
alpha = 0.1
df, pp_dict, pi_dict = train_pipeline(
       df=fire_df_post,
       factor1=factor1,
       factor2=factor2,
       target=target,
       covariates=covariates,
       pi_covariates=pi_covariates,
       alpha=alpha
)
ratio = 3.0  # Elliptical ratio between major and minor axis
df = add_elliptical_propagation_speed_to_df(df, ratio=ratio)
df = add_base_model_predictions_to_df(df, factor1, factor2, target, alpha)

# --------- Tests ---------
time_cols = [
 'start_datetime',
 'arrival_datetime_inc',
 'control_datetime',
 'extinction_datetime',
 'start_day'
]


# --- Pytest-wrapped tests (moved from module-level calls) ---

def test_preprocessing_pipeline():
    assert_df_from_file(
        df=fire_df,
        filename=DATA_PROCESSED + "/data_workflow/2_2_incendios_2014-2025.csv",
        time_cols=time_cols,
        exclude_cols=['control_datetime', 'start_time']
    )

def test_feature_engineering_pipeline():
    assert_df_from_file(
        df=fire_df_fe, 
        filename=DATA_PROCESSED + "/data_workflow/2_3_incendios_2014-2025.csv", 
        time_cols=time_cols, 
        exclude_cols=['control_datetime', 'start_time', "weather_index", "weather_index_full", "initial_fuel" ,"initial_fuel_reduced"]
    )

def test_postprocessing_pipeline():
    assert_df_from_file(
        df=fire_df_post, 
        filename=DATA_PROCESSED + "/data_workflow/3_0_incendios_2014-2025.csv", 
        time_cols=time_cols, 
        exclude_cols=['control_datetime', 'start_time', "weather_index", "weather_index_full", "initial_fuel_reduced", "split2", "split3", "zone_alert"]
    )

def test_predictions_matches():
    assert_predictions_match(
        df=df[df['fire_id'] == 215011234],  
        pp_dict=pp_dict,
        covariates=covariates,
        factor1=factor1,
        factor2=factor2
    )
