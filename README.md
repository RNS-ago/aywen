<h1 align="center">
<img src="docs/logo_transparent.png" alt="Aywen Logo" width="300">
</h1><br>

Aywen is a lightweight toolkit for preprocessing, validating, and feature‑engineering wildfire incident and dispatch datasets. It provides a reproducible pipeline to load multiple CSVs, normalize and match incidents with dispatches, and generate QA/QC artifacts. It also includes geospatial helpers to attach zone metadata from shapefiles.

## Requirements
- Python: >= 3.11
- Dependencies (installed automatically):
  - `pandas>=2.3.1`
  - `numpy>=2.3.2`
  - `geopandas>=1.1.1`
  
Note: As of version 0.1.0, `geopandas` is a required dependency and is installed by default.

## Installation

### Using pip (local repo)
Create and activate a virtual environment, then install in editable mode. All dependencies from `pyproject.toml` will be installed automatically.

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

macOS/Linux (bash):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Using uv
This project includes a `uv.lock` for reproducible environments.

```bash
# Create and activate a virtualenv managed by uv
uv venv
source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1

# Sync dependencies from pyproject/uv.lock
uv sync

# Alternatively, perform an editable install
uv pip install -e .
```

## Quick Start

### Preprocessing pipeline
Load multiple CSVs (incidents + dispatches), normalize columns, standardize IDs, match, and generate QA/QC outputs.

```python
from aywen.preprocessing import preprocess_pipeline

fire_csvs = [
    "data/incendios_2019.csv",
    "data/incendios_2020.csv",
]
dispatch_csvs = [
    "data/despachos_2019.csv",
    "data/despachos_2020.csv",
]

fire_df, dispatch_df, qaqc_df = preprocess_pipeline(
    fire_csvs,
    dispatch_csvs,
    output_dir="out",  # optional, saves CSVs
)
```

### Geospatial features
Attach zone labels from a shapefile to rows with lat/lon.

```python
from aywen.features import add_zones_to_df

gdf = add_zones_to_df(
    df=fire_df,
    shapefile_path="data/zones.shp",
    lon_col="longitude",
    lat_col="latitude",
    zone_col="area",
    new_zone_col="zone_alert",
)
```

## Notes
- Expected columns referenced by helpers (e.g., `hr_arribo`, `glosa`, `recurso`, `fire_id`, `arrival_datetime_inc`) are standardized inside the pipeline; see `src/aywen/preprocessing.py` and `src/aywen/features.py` for details.
- Logging is preconfigured with colorized console output; see `src/aywen/logging_setup.py`.

## Development
- Versioning and dependencies are defined in `pyproject.toml`. Use `uv sync` for a deterministic environment (via `uv.lock`), or `pip install -e .` for editable development.
- Contributions welcome. Please open an issue or PR with a clear description and minimal reproducible example.
