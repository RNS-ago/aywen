from __future__ import annotations
from typing import Any, Dict, Optional, List, Mapping, Union, Iterable
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import shutil
import mlflow
import logging
import os
from pathlib import Path
import ast
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse, unquote
from sklearn.base import BaseEstimator, TransformerMixin



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



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

def save_artifacts(
    artifacts_dir: Union[str, Path],
    df: Optional[pd.DataFrame] = None,
    model_dict: Optional[Dict[str, Any]] = None,
    pi_dict: Optional[Dict[str, Any]] = None,
    factors: Optional[List[str]] = None,
    covariates: Optional[List[str]] = None,
    covariates_categorical: Optional[List[str]] = None,
    pi_covariates: Optional[List[str]] = None,
    fuel_mapping: Optional[Mapping[Any, Any]] = None,
    target: Optional[str] = None,
    alpha: Optional[float] = None,
    ratio: Optional[float] = None,
    metrics: Optional[pd.DataFrame] = None,
    # extra, arbitrary outputs to save; keys are filenames (optional),
    # values are dict | list[dict] | DataFrame
    outputs: Optional[Dict[Optional[str], Any]] = None,
    # filenames (override defaults if you want)
    df_name: str = "data.parquet",
    model_name: str = "model.pkl",
    pi_name: str = "pi.json",
    fuel_mapping_name: str = "fuel_mapping.json",
    meta_name: str = "meta.json",
    schema_name: str = "schema.json",
    metrics_name: str = "metrics.csv",
    log_to_mlflow: bool = True,
) -> Dict[str, str]:
    """
    Save run artifacts to `artifacts_dir` and (optionally) log them to MLflow.

    - df -> parquet + schema.json + 50-row CSV preview
    - model_dict -> joblib pickle
    - pi_dict, fuel_mapping -> JSON using `serialize`
    - meta -> JSON constructed from provided metadata
    - metrics -> CSV
    - outputs -> extra arbitrary artifacts; if key (filename) is None, a default is inferred

    Returns a dict of artifact logical names -> absolute paths.
    """
    

    paths: Dict[str, str] = {}

    adir = Path(artifacts_dir)
    if adir.exists():
        shutil.rmtree(adir)  # clean folder before writing new artifacts
    adir.mkdir(parents=True, exist_ok=True)

    # ---------------- helpers ----------------
    def _save_one(data: Any, name: Optional[str] = None) -> Path:
        """
        Save `data` by type:
          - dict -> JSON
          - list[dict] -> JSON array
          - DataFrame -> Parquet (+ schema.json & head CSV preview)
        If `name` is None, infer a default.
        """
        # infer default filename if not provided
        if name is None:
            if isinstance(data, dict):
                name = "output.json"
            elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
                name = "output.json"
            elif isinstance(data, pd.DataFrame):
                name = "output.parquet"
            else:
                raise TypeError(
                    "Unsupported data without explicit filename. "
                    "Must be dict, list[dict], or pandas DataFrame."
                )

        path = adir / name

        if isinstance(data, dict):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, cls=NpEncoder)
        elif isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise ValueError("List must contain only dicts to be saved as JSON.")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, cls=NpEncoder)
        elif isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
            # schema for the dataframe we just saved
            try:
                mgr = DtypeManager.from_df(data)
                mgr.save(adir / "schema.json")
            except Exception as e:
                logger.warning("Could not save schema.json: %s", e)

            # quick preview
            preview_path = adir / name.replace(".parquet", "_preview.csv")
            try:
                data.head(50).to_csv(preview_path, index=False)
            except Exception:
                pass
        else:
            raise TypeError(
                f"Unsupported type {type(data).__name__}; expected dict, list[dict], or DataFrame."
            )

        return path

    # ---------------- main artifacts ----------------
    # data (parquet + schema + preview)
    if df is not None:
        # If you want schema of only covariates (as your earlier function did),
        # you can additionally write that specific schema file:
        try:
            # save main data
            ext = os.path.splitext(df_name)[1].lower()
            if ext == ".json":
                df = df.to_dict(orient="records")
            p = _save_one(df, df_name)
            paths["data"] = str(p.resolve())
            logger.info("Saved data   -> %s", p)
            # optional: schema of covariates
            if covariates:
                mgr_cov = DtypeManager.from_df(df[covariates])
                mgr_cov.save(adir / schema_name)
                paths["schema"] = str((adir / schema_name).resolve())
                logger.info("Saved schema -> %s", adir / schema_name)
        except Exception as e:
            logger.exception("Failed saving data: %s", e)

    # model
    if model_dict is not None:
        model_path = adir / model_name
        joblib.dump(model_dict, model_path)
        paths["model"] = str(model_path.resolve())
        logger.info("Saved model  -> %s", model_path)

    # predictive intervals (use your serialize)
    if pi_dict is not None:
        try:
            pi_serialized = serialize(pi_dict)
        except Exception:
            # assume already serialized
            pi_serialized = pi_dict
        p = _save_one(pi_serialized, pi_name)
        paths["pi"] = str(p.resolve())
        logger.info("Saved PI     -> %s", p)

    # fuel mapping (use your serialize)
    if fuel_mapping is not None:
        try:
            fuel_serialized = serialize(fuel_mapping)
        except Exception:
            fuel_serialized = fuel_mapping
        p = _save_one(fuel_serialized, fuel_mapping_name)
        paths["fuel_mapping"] = str(p.resolve())
        logger.info("Saved fuel   -> %s", p)

    # metadata
    meta = {
        "factors": factors,
        "covariates": covariates,
        "covariates_categorical": covariates_categorical,
        "pi_covariates": pi_covariates,
        "target": target,
        "alpha": alpha,
        "ratio": ratio,
    }
    # drop Nones to keep it clean
    meta = {k: v for k, v in meta.items() if v is not None}
    if meta:
        p = _save_one(meta, meta_name)
        paths["meta"] = str(p.resolve())
        logger.info("Saved meta   -> %s", p)

    # metrics
    if metrics is not None:
        mpath = adir / metrics_name
        metrics.to_csv(mpath, index=False)
        paths["metrics"] = str(mpath.resolve())
        logger.info("Saved metrics-> %s", mpath)

    # ---------------- extra outputs ----------------
    if outputs:
        for name, data in outputs.items():
            p = _save_one(data, name)  # name may be None -> inferred
            logical = f"output:{name or Path(p).name}"
            paths[logical] = str(p.resolve())
            logger.info("Saved extra  -> %s", p)

    # ---------------- MLflow log ----------------
    if log_to_mlflow:
        try:
            mlflow.log_artifacts(str(adir))
        except Exception as e:
            logger.warning("MLflow log_artifacts failed: %s", e)

    return paths

# ------- loading artifacts -------


def _to_local_path(uri: str) -> Path:
    """
    Convert an MLflow file:// URI to a local Path.
    Works for artifact_location and run.info.artifact_uri.
    """
    p = urlparse(uri)
    if p.scheme and p.scheme != "file":
        raise ValueError(f"Non-local artifact URI not supported: {uri}")
    return Path(unquote(p.path.lstrip("/")))


def load_artifacts(
    run_id: str,
    experiment_name: str,
    *,
    # filenames used by your save_artifacts()
    df_name: str = "data.parquet",
    model_name: str = "model.pkl",
    pi_name: str = "pi.json",
    fuel_mapping_name: str = "fuel_mapping.json",
    meta_name: str = "meta.json",
    schema_name: str = "schema.json",
    metrics_name: str = "metrics.csv",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load artifacts for a local MLflow run (no temp dirs, no remote backends).

    Returns:
      {
        'paths': {...},
        'df': DataFrame|None,
        'schema': DtypeManager|dict|None,
        'model': Any|None,
        'pi': dict|None,
        'fuel_mapping': dict|None,
        'meta': dict|None,
        'metrics': DataFrame|None,
        'artifact_root': Path
      }
    """
    client = MlflowClient()

    # Validate experiment & run
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name!r} not found.")
    run = client.get_run(run_id)
    if run.info.experiment_id != exp.experiment_id:
        raise ValueError(
            f"Run {run_id!r} belongs to experiment_id={run.info.experiment_id}, "
            f"not {experiment_name!r} (id={exp.experiment_id})."
        )

    # Resolve local artifact directory for the run
    # Typical layout: <exp_artifact_location>/<run_id>/artifacts/
    run_art_root = _to_local_path(run.info.artifact_uri)

    out: Dict[str, Any] = {
        "paths": {},
        "df": None,
        "schema": None,
        "model": None,
        "pi": None,
        "fuel_mapping": None,
        "meta": None,
        "metrics": None,
        "artifact_root": run_art_root,
    }

    def _read_json(path: Path) -> Optional[dict]:
        if not path.exists():
            if strict: raise FileNotFoundError(f"Missing artifact: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        out["paths"][path.name] = str(path.resolve())
        return obj

    def _maybe(path: Path) -> bool:
        if path.exists():
            out["paths"][path.name] = str(path.resolve())
            return True
        if strict:
            raise FileNotFoundError(f"Missing artifact: {path}")
        return False

    # Load core artifacts (best-effort unless strict=True)
    df_path = run_art_root / df_name
    if _maybe(df_path):
        out["df"] = pd.read_parquet(df_path)

    schema_path = run_art_root / schema_name
    if schema_path.exists():
        try:
            out["schema"] = DtypeManager.load(schema_path)
        except Exception:
            out["schema"] = _read_json(schema_path)

    model_path = run_art_root / model_name
    if _maybe(model_path):
        out["model"] = joblib.load(model_path)

    pi_path = run_art_root / pi_name
    if pi_path.exists():
        raw = _read_json(pi_path)
        if raw is not None:
            try:
                out["pi"] = restore(raw)
            except Exception:
                out["pi"] = raw

    fuel_path = run_art_root / fuel_mapping_name
    if fuel_path.exists():
        raw = _read_json(fuel_path)
        if raw is not None:
            try:
                out["fuel_mapping"] = restore(raw)
            except Exception:
                out["fuel_mapping"] = raw

    meta_path = run_art_root / meta_name
    if meta_path.exists():
        out["meta"] = _read_json(meta_path)

    metrics_path = run_art_root / metrics_name
    if metrics_path.exists():
        out["metrics"] = pd.read_csv(metrics_path)
        out["paths"][metrics_path.name] = str(metrics_path.resolve())

    return out




def _load_artifacts(run_id: str, experiment_name: str) -> Dict[str, Any]:
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



def _load_latest_run(experiment_name, run_name=None):
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



def _escape(s: str) -> str:
    return s.replace("'", "''")

def _search_runs_paginated(client: MlflowClient, exp_id: str, filter_string: str,
                           order_by: List[str], page_size: int = 200, max_pages: int = 50):
    # MlflowClient.search_runs already paginates internally, but some versions cap hard.
    # This helper just lets us request a bigger page and stop early when we find something.
    return client.search_runs(
        [exp_id],
        filter_string=filter_string,
        order_by=order_by,
        max_results=page_size,
    )

def load_latest_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    *,
    status: Optional[str] = "FINISHED",   # "FINISHED" | "FAILED" | "RUNNING" | None (no status filter)
    only_active: bool = True,
):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name!r} not found.")

    logger.info("Using experiment: %s | id: %s | at: %s",
                exp.name, exp.experiment_id, exp.artifact_location)

    order_by = ["attributes.start_time DESC"]

    def _build_filter(use_attr_run_name: bool, use_status: bool) -> str:
        parts = []
        if use_status and status:
            parts.append(f"attributes.status = '{_escape(status)}'")
        if run_name:
            if use_attr_run_name:
                # newer MLflow supports attributes.run_name
                parts.append(f"attributes.run_name = '{_escape(run_name)}'")
            else:
                # fallback: name is stored as tag
                parts.append(f"tags.mlflow.runName = '{_escape(run_name)}'")
        return " and ".join(parts) if parts else ""

    # Try 1: attributes.run_name + (optional) status
    runs = _search_runs_paginated(client, exp.experiment_id,
                                  _build_filter(use_attr_run_name=True, use_status=True),
                                  order_by)
    # If none, Try 2: tag filter + (optional) status
    if not runs:
        runs = _search_runs_paginated(client, exp.experiment_id,
                                      _build_filter(use_attr_run_name=False, use_status=True),
                                      order_by)
    # If still none and we required FINISHED, Try 3: drop status filter
    if not runs and status:
        runs = _search_runs_paginated(client, exp.experiment_id,
                                      _build_filter(use_attr_run_name=True, use_status=False),
                                      order_by)
        if not runs:
            runs = _search_runs_paginated(client, exp.experiment_id,
                                          _build_filter(use_attr_run_name=False, use_status=False),
                                          order_by)

    # Filter out deleted client-side if requested
    if only_active:
        runs = [r for r in runs if r.info.lifecycle_stage == "active"]

    if not runs:
        rn = f" with run_name={run_name!r}" if run_name else ""
        # Quick diagnostic to help you see what's there
        diag = client.search_runs([exp.experiment_id],
                                  filter_string="",
                                  order_by=["attributes.start_time DESC"],
                                  max_results=10)
        diag_lines = [
            f"- {r.info.run_id} | name={r.data.tags.get('mlflow.runName')} | "
            f"attr_name={getattr(r.data.tags, 'run_name', None)} | status={r.info.status} | stage={r.info.lifecycle_stage}"
            for r in diag
        ]
        raise ValueError(
            f"No runs found in experiment {experiment_name!r}{rn} with status={status!r}.\n"
            "Here are the 10 most recent runs:\n" + "\n".join(diag_lines)
        )

    run = runs[0]
    logger.info("Selected run %s (name=%s, status=%s, stage=%s)",
                run.info.run_id, run.data.tags.get("mlflow.runName"),
                run.info.status, run.info.lifecycle_stage)

    return load_artifacts(run.info.run_id, experiment_name)





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

