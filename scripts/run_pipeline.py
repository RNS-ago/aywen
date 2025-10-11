import os
import json
import time
import logging
from pathlib import Path

import mlflow
import shutil
import argparse

from aywen.logging_setup import configure_logging
from aywen.preprocessing import preprocessing_pipeline
from aywen.fire_features import feature_engineering_pipeline
from aywen.postprocessing import postprocessing_pipeline, COVARIATES, COVARIATES_CATEGORICAL
from aywen.fire_features import DEFAULT_COLUMNS_DICT
from aywen.training import train_pipeline, add_elliptical_propagation_speed_to_df, add_base_model_predictions_to_df, compute_residual_diagnostics
from aywen.utils import prepare_long_df, save_artifacts

# ---------------- paths  ----------------
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

# ------ retrieve default columns and values ------
factors = DEFAULT_COLUMNS_DICT["factors"]  # ["zone_WE", "zone_NS"]
covariates = COVARIATES
covariates_categorical = COVARIATES_CATEGORICAL
target = "propagation_speed_mm"
pi_covariates = ["diurnal_nocturnal", 'high_season']

factor1 = factors[0]  # "zone_WE"
factor2 = factors[1]  # "zone_NS"
alpha = 0.1  # 80% prediction interval
ratio = 3.0  # Elliptical ratio between major and minor axis (currently unused in this file)

def main():
    
    ap = argparse.ArgumentParser(description="Scorer")
    ap.add_argument("--store_path", default=r"C:\Users\lopez\GitRepos\aywen\scripts\mlruns", type=Path)
    ap.add_argument("--experiment-name", default= "arauco_fire_pipeline", type=str)
    ap.add_argument("--run-name", default=None, type=str)
    ap.add_argument("--kitral-fuel", action='store_true', help="Use kitral fuel values instead of default fuel values")
    args = ap.parse_args()

    if args.run_name is None:
        args.run_name = "kitral" if args.kitral_fuel else "default" + "-run"
    
    # --- logging setup ---
    configure_logging()
    root = logging.getLogger()
    root.setLevel(logging.IMPORTANT)   # switches “mode” to IMPORTANT
    logger = logging.getLogger("aywen_logger")
    logger.handlers.clear()            # child has no handlers
    logger.propagate = True            # let root handle
    logger.important("Starting full pipeline")

    # --- MLflow setup ---
    mlflow.set_tracking_uri(args.store_path.as_uri())
    mlflow.set_experiment(args.experiment_name)
    start_ts = time.time()

    # --- Main pipeline ---
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: experiment='{args.experiment_name}', run_id='{run_id}'")

        # ---- Log high-level parameters  ----
        mlflow.log_param("factor1", factor1)
        mlflow.log_param("factor2", factor2)
        mlflow.log_param("target", target)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("ratio", ratio)
        mlflow.log_param("n_covariates", len(covariates))
        mlflow.log_param("n_pi_covariates", len(pi_covariates))
        mlflow.log_param("covariates_json", json.dumps(covariates))
        mlflow.log_param("pi_covariates_json", json.dumps(pi_covariates))

        # --------- preprocessing pipeline ---------
        fire_df, _ = preprocessing_pipeline(FIRE_CSVS, DISPATCH_CSVS)
        mlflow.log_metric("df_rows_after_preprocessing", len(fire_df))

        # --------- feature engineering pipeline ---------
        fire_df = feature_engineering_pipeline(
            fire_df,
            zone_shapefile_path=ZONE_SHAPEFILE_PATH,
            topo_csv_path=TOPO_CSV_PATH,
            fuel_tiff_path=TIFF_FUEL_PATH,
            kitral_fuel=args.kitral_fuel,
        )
        mlflow.log_metric("df_rows_after_features", len(fire_df))

        # --------- postprocessing pipeline ---------
        fire_df, fuel_mapping = postprocessing_pipeline(
            fire_df,
            kitral_fuel=args.kitral_fuel
        )
        mlflow.log_metric("df_rows_after_postprocessing", len(fire_df))

        # --------- training pipeline ---------
        fire_df, pp_dict, pi_dict = train_pipeline(
            df=fire_df,
            factor1=factor1,
            factor2=factor2,
            target=target,
            covariates=covariates,
            pi_covariates=pi_covariates,
            alpha=alpha
        )
        mlflow.log_metric("df_final_rows", len(fire_df))

        # Add elliptical propagation speed and base model predictions to df
        fire_df = add_elliptical_propagation_speed_to_df(fire_df, ratio=ratio)
        fire_df = add_base_model_predictions_to_df(fire_df, factor1, factor2, target, alpha)

        # residual metrics
        df_long = prepare_long_df(fire_df)
        metrics = compute_residual_diagnostics(df_long, cols= ['model', 'split2'], y_true=target, y_pred='prediction')
        mlflow.log_metric("MAE_base_train", metrics.loc[('base', 'train+valid'), 'MAE'])
        mlflow.log_metric("MAE_base_test", metrics.loc[('base', 'test'), 'MAE'])
        mlflow.log_metric("MAE_xgb0_train", metrics.loc[('xgb0', 'train+valid'), 'MAE'])
        mlflow.log_metric("MAE_xgb0_test", metrics.loc[('xgb0', 'test'), 'MAE'])
        mlflow.log_metric("MAE_xgb_train", metrics.loc[('xgb', 'train+valid'), 'MAE'])
        mlflow.log_metric("MAE_xgb_test", metrics.loc[('xgb', 'test'), 'MAE'])

        # artifacts directory
        artifacts_dir = Path("artifacts")

        # clean folder before writing new artifacts
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        # save outputs locally
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Save artifacts
        save_artifacts(
            artifacts_dir,
            fire_df,
            pp_dict,
            pi_dict,
            factors,
            covariates,
            covariates_categorical,
            pi_covariates,
            fuel_mapping,
            target,
            alpha,
            ratio
        )

        # Log entire directory so you also capture both PKLs and the DF
        mlflow.log_artifacts(str(artifacts_dir))

        # Quick preview CSV for convenience, limited to 50 rows
        preview_path = artifacts_dir / "final_df_preview.csv"
        try:
            fire_df.head(50).to_csv(preview_path, index=False)
            mlflow.log_artifact(str(preview_path))
        except Exception:
            pass

        # Timing
        total_seconds = time.time() - start_ts
        mlflow.log_metric("total_runtime_seconds", total_seconds)
        logger.info(f"MLflow run finished in {total_seconds:.1f}s — run_id='{run_id}'")

if __name__ == "__main__":
    main()
