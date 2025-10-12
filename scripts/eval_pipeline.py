import os
import logging
import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import mlflow
import shutil

from aywen.logging_setup import configure_logging
from aywen.fire_features import feature_engineering_pipeline, DEFAULT_COLUMNS_DICT
from aywen.postprocessing import add_groupwise_mapping
from aywen.utils import load_latest_run, DtypeManager
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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def _save_output(data: pd.DataFrame, artifacts_dir: Path, name: str = None) -> None:
    """
    Save data to file.
    - dict -> JSON object
    - list of dicts -> JSON array
    - DataFrame -> CSV
    """

    if name is None:
        if isinstance(data, (dict,list)):
            name = "output.json"
            data = data.to_dict(orient="records")
        elif isinstance(data, pd.DataFrame):
            name = "output.parquet"
        else:
            raise TypeError(
                "Data must be dict, list of dicts, or pandas DataFrame, "
                f"not {type(data).__name__}"
            )
        

    path = artifacts_dir / name
    if isinstance(data, dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=NpEncoder)
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2,cls=NpEncoder)
        else:
            raise ValueError("List must contain only dicts to be saved as JSON")
    elif isinstance(data, pd.DataFrame):
        data.to_parquet(path, index=False)
        mgr = DtypeManager.from_df(data)
        mgr.save(artifacts_dir / "schema.json")
        # Quick preview CSV for convenience, limited to 50 rows
        preview_path = artifacts_dir / name.replace(".parquet", "_preview.csv")
        try:
            data.head(50).to_csv(preview_path, index=False)
            mlflow.log_artifact(str(preview_path))
        except Exception:
            pass

    else:
        raise TypeError(
            "Data must be dict, list of dicts, or pandas DataFrame, "
            f"not {type(data).__name__}"
        )
    
    mlflow.log_artifacts(str(artifacts_dir))
    
    
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
    ap.add_argument("--output-run-name", default=RUN_NAME, type=str)
    args = ap.parse_args()

    # --- logging setup ---
    configure_logging()
    logger = logging.getLogger("aywen_logger")
    logger.setLevel(logging.INFO)
    logger.info('Inicio evaluacion de modelo')
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

    with mlflow.start_run(run_name=args.output_run_name) as run:
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


        # artifacts directory
        artifacts_dir = Path("artifacts")
        # clean folder before writing new artifacts
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        # save outputs locally
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Save artifacts
        _save_output(out, artifacts_dir, args.output)
        logger.info(f"Saved artifacts to {artifacts_dir}")
    
    
if __name__ == "__main__":
    main()
    