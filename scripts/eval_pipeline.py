import os
import logging
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from aywen.logging_setup import configure_logging
from aywen.fire_features import feature_engineering_pipeline, DEFAULT_COLUMNS_DICT
from aywen.postprocessing import add_groupwise_mapping
from aywen.utils import load_latest_run, save_artifacts
from aywen.evaluating import evaluation_pipeline


# ---------------- paths  ----------------
# Paths to data files
DIR_ROOT =  "G:/Shared drives/OpturionHome/AraucoFire"
DATA_ORIGINAL = DIR_ROOT + "/2_data/original"
DATA_PROCESSED = DIR_ROOT + "/2_data/processed"
ZONE_SHAPEFILE_PATH = DATA_ORIGINAL + "/tiles_arauco/Zonas_alerta/Zonas_alerta_.shp"
TOPO_CSV_PATH = DATA_PROCESSED + "/topographic_data.csv"
FUEL_TIFF_PATH = DATA_ORIGINAL + "/tiles_arauco/ModeloCombustible202409/ModeloCombustible202409.tif"


# ------ retrieve default columns and values ------
factors = DEFAULT_COLUMNS_DICT["factors"]  # ["zone_WE", "zone_NS"]
factor1 = factors[0]  # "zone_WE"
factor2 = factors[1]  # "zone_NS"

def _load_input_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize to list of dicts
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("JSON input must be an object or a list of objects")

        return pd.DataFrame(data)

    elif ext == ".csv":
        return pd.read_csv(path, encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# -------- Config resolved from env, with sensible defaults --------
EXPERIMENT_NAME_INPUT = "arauco_fire_pipeline"
EXPERIMENT_NAME = "arauco_fire_evaluation"
RUN_NAME        = "default-run"
DEFAULT_TRACKING_DIR = r"C:\Users\lopez\GitRepos\aywen\scripts\mlruns"

def main():

    ap = argparse.ArgumentParser(description="Scorer")
    ap.add_argument("--store_path", default=DEFAULT_TRACKING_DIR, type=Path)
    ap.add_argument("--input", default=r"G:\Shared drives\OpturionHome\AraucoFire\2_data\processed\in_mlruns\input.json", type=Path)
    ap.add_argument("--output", default=None, type=Path)
    ap.add_argument("--experiment-name", default=EXPERIMENT_NAME, type=str)
    ap.add_argument("--input-run-name", default=RUN_NAME, type=str)
    ap.add_argument("--run-name", default=RUN_NAME, type=str)
    args = ap.parse_args()

    if args.output is None:
        ext = os.path.splitext(args.input.name)[1]
        args.output = "output" + ext

    if args.run_name is None:
        args.run_name = args.input_run_name

    # --- logging setup ---
    configure_logging()
    root = logging.getLogger()
    root.setLevel(logging.IMPORTANT)   # switches “mode” to IMPORTANT
    logger = logging.getLogger("aywen_logger")
    logger.handlers.clear()            # child has no handlers
    logger.propagate = True            # let root handle
    logger.important("Starting full pipeline")
    # quiet rasterio
    for name in ("rasterio", "rasterio.env", "rasterio._env"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)   # or CRITICAL
        lg.propagate = False

    # ------- load artifacts from latest run -------
    mlflow.set_tracking_uri(args.store_path.resolve().as_uri())
    artifacts = load_latest_run(EXPERIMENT_NAME_INPUT, run_name=args.input_run_name)
    mgr = artifacts['schema']
    pp = artifacts['model']
    pi = artifacts['pi']
    meta = artifacts['meta']
    fuel_mapping = artifacts['fuel_mapping']

    # ------ load input as df ------
    df = _load_input_file(args.input)
    logging.info(f"Loaded {len(df)} samples from {args.input}")

    # --- Select/create experiment and start run ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: experiment='{EXPERIMENT_NAME}', run_id='{run_id}'")
    
        # ------- feature engineering pipelione --------
        out = df.copy()
        out = feature_engineering_pipeline(
            out, 
            zone_shapefile_path=ZONE_SHAPEFILE_PATH, 
            zone_threshold=0,  # Do not drop any zones
            topo_csv_path=TOPO_CSV_PATH, 
            fuel_tiff_path=FUEL_TIFF_PATH, 
            kitral_fuel=True,  # use kitral fuel values read from tiff
            skip_weather=True, # no weather values, they will be loaded from API
            skip_target=True # no target column in input, it will be predicted
        )

        # ------ map fuel values to reduced fuel values ------
        out = add_groupwise_mapping(
            df=out,
            mapping=fuel_mapping,
            source_col="initial_fuel", # default naming 
            target_col="initial_fuel_reduced" # default naming
        )

        # ------ evaluation pipeline ------
        out = evaluation_pipeline(
            models=pp,
            pi=pi,
            df=out,
            factor1=factor1,
            factor2=factor2,
            mgr=mgr,
            pi_covariates=meta["pi_covariates"],
            ratio=meta["ratio"],
            prediction_col = "prediction_circular_speed_mm",
            lo_col = "lo_circular_speed_mm",
            hi_col = "hi_circular_speed_mm"
        ) #  elliptical speed columns names to default names

        # ------- save artifacts -------
        save_artifacts(artifacts_dir="artifacts", df=out, df_name=args.output)
    
    
if __name__ == "__main__":
    main()
    