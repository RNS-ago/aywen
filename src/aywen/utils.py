import pandas as pd
import logging
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable
import ast
from mlflow.tracking import MlflowClient
import joblib
from urllib.parse import urlparse, unquote



# # --- Configure logging globally ---
# logging.basicConfig(
#     level=logging.INFO,  # Default level (can be changed later)
#     format="%(message)s"  # Only show the message (no timestamps/levels unless you want them)
# )

logger = logging.getLogger("aywen_logger")


def concatnate_name_from_paths(paths: list[str]) -> str:
    report_type = os.path.splitext(os.path.basename(paths[0]))[0].split('_')[0]
    
    report_date_range = []
    for path in paths:
        report_date_range.append(os.path.splitext(os.path.basename(path))[0].split('_')[1])
        
    report_date_range = '_'.join(report_date_range)
    
    return f"{report_type}_{report_date_range}"
    
    
def assert_time_diff(df1, df2, key1, key2, type='equal'):

    left = df1[['fire_id', key1]].copy()
    right = df2[['fire_id', key2]].copy()

    # cast to datetime
    left[key1] = pd.to_datetime(left[key1])
    right[key2] = pd.to_datetime(right[key2])

    # group by fire_id and get min
    right = right.groupby('fire_id').min().reset_index()
    #  merge
    merged = pd.merge(left, right, on='fire_id', how='left')

    # error
    merged['time_diff'] = (merged[key1] - merged[key2]).dt.total_seconds() / 60
    merged.dropna(subset=['time_diff'], inplace=True)

    if type == 'equal':
        assert np.all(merged['time_diff'] == 0), "Error: There are records with non-zero time difference"
    elif type == 'positive':
        assert np.all(merged['time_diff'] >= 0), "Error: There are records with non-positive time difference"
    elif type == 'negative':
        assert np.all(merged['time_diff'] <= 0), "Error: There are records with non-negative time difference"
        




class SpeedCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to categorize propagation speed into ordinal bins,
    with optional visualization support.
    """
    def __init__(self, 
                 bins=None, 
                 labels=None, 
                 right=False):
        self.bins = bins if bins is not None else [0, 1.7, 10, 33, 83, np.inf] # meters / minute
        self.labels = labels if labels is not None else ['baja', 'media', 'alta', 'muy alta', 'extrema']
        self.right = right
   
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()    
        X_cat = pd.cut(
            X,
            bins=self.bins,
            labels=self.labels,
            right=self.right
        )
        return X_cat


class TimeOfDayFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, period=24):
        """
        Parameters:
        - period: periodicity of the cycle (default: 24 for hours)
        """
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform input hours into sine and cosine components with phase shift.
        """

        X = np.asarray(X).reshape(-1)
        theta = 2 * np.pi * X / self.period

        cos_hour = np.cos(theta)
        sin_hour = np.sin(theta)

        return sin_hour, cos_hour
    
    
class MonthCycleFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, period=12, peak_month=1):
        """
        period: number of months in a year (default 12)
        peak_month: the month (1–12) where the sine cycle should peak
        """
        self.period = period
        self.phase = 2 * np.pi * (peak_month - 1) / period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: array-like of months [1–12]
        Returns: DataFrame with sin_month and cos_month
        """
        X = np.asarray(X).reshape(-1)
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Input must be integer months in [1, 12].")
        theta = 2 * np.pi * (X - 1) / self.period
        
        return np.sin(theta - self.phase), np.cos(theta - self.phase)


# ------- dtypes functions -------


class DtypeManager:
    """
    Manage (persist + apply) pandas dtypes, including CategoricalDtype metadata.
    """

    def __repr__(self):
        return json.dumps(self.to_json_ready(), indent=2)

    # ---------- Constructors ----------
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DtypeManager":
        """Capture dtypes (including categorical categories & order) from df."""
        dtype_map: Dict[str, Any] = {}
        for col in df.columns:
            dt = df[col].dtype
            if isinstance(dt, pd.CategoricalDtype):
                dtype_map[col] = pd.CategoricalDtype(
                    categories=dt.categories,
                    ordered=dt.ordered
                )
            else:
                dtype_map[col] = dt
        return cls(dtype_map)

    @classmethod
    def load(cls, path: str | Path) -> "DtypeManager":
        """Load from a JSON file produced by save()."""
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return cls(cls._deserialize(data))

    # ---------- Core ----------
    def __init__(self, dtype_map: Dict[str, Any]):
        self.dtype_map = dtype_map  # values: np.dtype or pd.CategoricalDtype

    def save(self, path: str | Path) -> None:
        """
        Save to JSON. If already serializable, dump as-is; otherwise serialize.
        """
        to_dump = (
            self.dtype_map if self._is_serializable(self.dtype_map)
            else self._serialize(self.dtype_map)
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_dump, f, indent=2)

    def apply(
        self,
        df: pd.DataFrame,
        *,
        strict: bool = False,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = [],
        copy: bool = True,
        drop_extras: bool = False,   # drop columns not in the schema
    ) -> pd.DataFrame:
        """
        Cast df to stored dtypes, align columns to the schema's order,
        and ensure the resulting DataFrame has the same columns and order
        as the schema (or the include/exclude-filtered subset).

        - strict=True: error if df is missing any required columns (after include/exclude).
        - include/exclude: limit which stored columns to use (order still follows schema).
        - drop_extras=True: drop columns not present in the schema subset.
        - fill_missing=True: add missing columns (if not strict) with NA/NaT and
          a null-friendly dtype.
        - copy=True: return a copy (default) or modify df in place.
        """
        target = df.copy() if copy else df

        # Determine the target column order from schema (respect include/exclude)
        ordered_cols = list(self._select_columns(include, exclude))

        # Strict check for missing columns
        missing = [c for c in ordered_cols if c not in target.columns]
        if strict and missing:
            raise KeyError(f"Missing columns in df: {missing}")

        if not strict and missing:
            exclude = exclude + missing
            ordered_cols = list(self._select_columns(include, exclude))

        # Build cast map for columns that exist now
        cast_map: Dict[str, Any] = {c: self.dtype_map[c] for c in ordered_cols if c in target.columns}

        # Cast (keeps categories & order for categoricals; unknowns become NaN)
        target = target.astype(cast_map, copy=False)

        # Reorder & optionally drop extras
        if drop_extras:
            target = target.reindex(columns=ordered_cols)
        else:
            # Keep extras, but move schema columns to the front in schema order
            extras = [c for c in target.columns if c not in ordered_cols]
            target = target.loc[:, ordered_cols + extras]

        return target

    # ---------- Helpers ----------
    @staticmethod
    def _is_serializable(dtype_map: Dict[str, Any]) -> bool:
        """True if dtype_map looks JSON-ready (i.e., dicts of simple types)."""
        def ok(v: Any) -> bool:
            if isinstance(v, dict):
                return "type" in v and (
                    (v["type"] == "other" and "dtype" in v) or
                    (v["type"] == "category" and "categories" in v and "ordered" in v)
                )
            return False
        return all(ok(v) for v in dtype_map.values())

    @staticmethod
    def _serialize(dtype_map: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for col, dt in dtype_map.items():
            if isinstance(dt, pd.CategoricalDtype):
                out[col] = {
                    "type": "category",
                    "categories": dt.categories.tolist(),
                    "ordered": bool(dt.ordered),
                }
            else:
                out[col] = {"type": "other", "dtype": str(np.dtype(dt))}
        return out

    @staticmethod
    def _deserialize(data: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for col, info in data.items():
            if info["type"] == "category":
                out[col] = pd.CategoricalDtype(
                    categories=info["categories"],
                    ordered=info["ordered"],
                )
            else:
                out[col] = np.dtype(info["dtype"])
        return out

    def _select_columns(
        self,
        include: Optional[Iterable[str]],
        exclude: Optional[Iterable[str]],
    ) -> Iterable[str]:
        cols = list(self.dtype_map.keys())
        if include is not None:
            inc = set(include)
            cols = [c for c in cols if c in inc]
        if exclude is not None:
            excl = set(exclude)
            cols = [c for c in cols if c not in excl]
        return cols

    @staticmethod
    def _make_missing_series(n: int, dtype: Any, index=None) -> pd.Series:
        """Create a length-n Series of NA/NaT with a dtype that can hold missing."""
        if isinstance(dtype, pd.CategoricalDtype):
            return pd.Series(
                pd.Categorical([pd.NA] * n, categories=dtype.categories, ordered=dtype.ordered),
                index=index,
                name=None
            )

        npdt = np.dtype(dtype)

        # datetime/timedelta -> NaT
        if npdt.kind in ("M", "m"):  # datetime64/timedelta64
            return pd.Series(pd.NaT, index=index, dtype=npdt)

        # floats can hold NaN directly
        if npdt.kind == "f":
            return pd.Series(np.nan, index=index, dtype=npdt)

        # booleans -> pandas nullable boolean
        if npdt.kind == "b":
            return pd.Series(pd.NA, index=index, dtype=pd.BooleanDtype())

        # integers/unsigned -> pandas nullable integer with matching width
        if npdt.kind in ("i", "u"):
            bits = npdt.itemsize * 8
            dtype_str = f"Int{bits}" if npdt.kind == "i" else f"UInt{bits}"
            return pd.Series(pd.NA, index=index, dtype=dtype_str)

        # string/object -> object with NA
        if npdt.kind in ("O", "U", "S"):
            return pd.Series(pd.NA, index=index, dtype="object")

        # fallback: object
        return pd.Series(pd.NA, index=index, dtype="object")

    # ---------- Convenience ----------
    def to_dict(self) -> Dict[str, Any]:
        """Return the in-memory dtype map (np.dtype / CategoricalDtype values)."""
        return self.dtype_map.copy()

    def to_json_ready(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation without writing to disk."""
        return self._serialize(self.dtype_map)
    
# ------- dictionary -------

def serialize(data: dict) -> dict:
    """
    Convert a dict with tuple keys into a JSON-serializable dict.
    Works recursively for nested dicts.
    """
    result = {}
    for k, v in data.items():
        # Convert tuple keys to string
        new_key = str(k) if isinstance(k, tuple) else k

        # If value is dict, recurse
        if isinstance(v, dict):
            result[new_key] = serialize(v)
        else:
            result[new_key] = v
    return result


def restore(data: dict) -> dict:
    """
    Restore a serialized dict back into its original form
    with tuple keys. Works recursively for nested dicts.
    """
    result = {}
    for k, v in data.items():
        try:
            # Try to interpret key as a tuple string
            new_key = ast.literal_eval(k) if isinstance(k, str) and k.startswith("(") else k
        except (ValueError, SyntaxError):
            new_key = k

        # If value is dict, recurse
        if isinstance(v, dict):
            result[new_key] = restore(v)
        else:
            result[new_key] = v
    return result


def replace_items(lst, replacements):
    """
    Replace items in a list if they exist in the replacements dictionary.

    Parameters:
        lst (list): Input list to modify.
        replacements (dict): Dictionary where key = old value, value = new value.

    Returns:
        list: New list with replacements applied.
    """
    return [replacements.get(item, item) for item in lst]



# ------ saving artifacts ------

def save_artifacts(
        artifacts_dir,
        df,
        pp_dict,
        pi_dict,
        factors,
        covariates,
        covariates_categorical,
        pi_covariates,
        fuel_mapping,
        target,
        alpha,
        ratio,
        metrics=None    
):
    df_path = f"{artifacts_dir}/data.parquet"
    schema_path = f"{artifacts_dir}/schema.json"
    model_path    = f"{artifacts_dir}/model.pkl"
    pi_path       = f"{artifacts_dir}/pi.json"
    fuel_mapping_path = f"{artifacts_dir}/fuel_mapping.json"
    meta_path     = f"{artifacts_dir}/meta.json"
    metrics_path  = f"{artifacts_dir}/metrics.csv"

    # data
    df.to_parquet(df_path, index=False)

    # manager schema     
    mgr = DtypeManager.from_df(df[covariates])
    mgr.save(schema_path)

    # model
    joblib.dump(pp_dict, model_path)

    # predictive intervals
    pi_serialized = serialize(pi_dict)
    with open(pi_path, "w", encoding="utf-8") as f:
        json.dump(pi_serialized, f, indent=2)

    # fuel mapping
    fuel_mapping_serialized = serialize(fuel_mapping)
    with open(fuel_mapping_path, "w", encoding="utf-8") as f:
        json.dump(fuel_mapping_serialized, f, indent=2)

    # metadata
    meta = {
        "factors": factors,
        "covariates": covariates,
        "covariates_categorical": covariates_categorical,
        "pi_covariates": pi_covariates,
        "target": target,
        "alpha": alpha,
        "ratio": ratio
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # metrics
    if metrics is not None:
        metrics.to_csv(metrics_path, index=False)

    # log
    logger.info("Saved data   -> %s", df_path)
    logger.info("Saved model -> %s", model_path)
    logger.info("Saved PI    -> %s", pi_path)
    logger.info("Saved meta  -> %s", meta_path)
    logger.info("Saved schema-> %s", schema_path)
    if metrics is not None:
        logger.info("Saved metrics-> %s", metrics_path)




def load_artifacts(run_id: str, experiment_name):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    # Experiment folder on disk (decode spaces etc.)
    exp_path = Path(unquote(urlparse(exp.artifact_location).path.lstrip("/")))
    run_dir = exp_path / run_id / "artifacts"

    artifacts = {}

    # DataFrame
    if (run_dir / "data.parquet").exists():
        artifacts["data"] = pd.read_parquet(run_dir / "data.parquet")
        logger.info("Loaded data")
    else:
        raise FileNotFoundError(f"Data file not found in run {run_dir / 'data.parquet'}")

    # Schema
    if (run_dir / "schema.json").exists():
        artifacts["schema"] = DtypeManager.load(run_dir / "schema.json")
        logger.info("Loaded schema")

    # Model
    if (run_dir / "model.pkl").exists():
        artifacts["model"] = joblib.load(run_dir / "model.pkl")
        logger.info("Loaded model")

    # Predictive intervals
    if (run_dir / "pi.json").exists():
        with open(run_dir / "pi.json", "r", encoding="utf-8") as f:
            artifacts["pi"] = restore(json.load(f))
        logger.info("Loaded predictive intervals")

    # Metadata
    if (run_dir / "meta.json").exists():
        with open(run_dir / "meta.json", "r", encoding="utf-8") as f:
            artifacts["meta"] = json.load(f)
        logger.info("Loaded metadata")

    # Fuel mapping
    if (run_dir / "fuel_mapping.json").exists():
        with open(run_dir / "fuel_mapping.json", "r", encoding="utf-8") as f:
            artifacts["fuel_mapping"] = restore(json.load(f))
        logger.info("Loaded fuel mapping")

    return artifacts



def load_latest_run(experiment_name, run_name=None):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    logger.info(
        "Using experiment: %s | id: %s | at: %s",
        exp.name,
        exp.experiment_id,
        exp.artifact_location,
    )

    # build filter string
    filter_string = "attributes.status = 'FINISHED'"
    if run_name is not None:
        filter_string += f" and tags.mlflow.runName = '{run_name}'"

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No finished runs found for experiment '{experiment_name}'"
            + (f" with run_name='{run_name}'" if run_name else "")
        )

    run_id = runs[0].info.run_id
    logger.info("Loading artifacts from latest run %s", run_id)
    return load_artifacts(run_id, experiment_name)



# ------- data frame utils -------

def prepare_long_df(df, cols=None, stubnames=None, i='fire_id', j='model', sep='_', suffix=r'\w+'):

    if cols is None:
        cols = [
            'fire_id', 
            'zone_WE', 
            'zone_NS',
            'high_season',
            'diurnal_nocturnal',
            'zone_alert',
            'split2', 
            'propagation_speed_mm',
            'prediction_xgb0','lo_xgb0', 'hi_xgb0',
            'prediction_xgb','lo_xgb', 'hi_xgb',
            'prediction_base', 'lo_base', 'hi_base'
        ]

        tmp = df[cols].copy()
        df_long = pd.wide_to_long(
            tmp, 
            stubnames=stubnames or ['prediction', 'lo', 'hi'], 
            i=i, 
            j=j, 
            sep=sep, 
            suffix=suffix
            ).reset_index()


        return df_long

