<h1 align="center">
<img src="docs/logo_transparent.png" alt="Aywen Logo" width="300">
</h1><br>

Aywen is a lightweight toolkit for preprocessing, validating, and feature‑engineering wildfire incident and dispatch datasets. It provides a reproducible pipeline to load multiple CSVs, normalize and match incidents with dispatches. It also includes geospatial helpers to attach zone metadata from shapefiles.   

## Requirements
- Python: >= 3.11
- Dependencies (installed automatically):
  - `pandas>=2.3.1`
  - `numpy>=2.3.2`
  - `geopandas>=1.1.1`
  - `scikit-learn>=1.7.1`
  - `tqdm>=4.67.1`
  - `richdem>=2.3.0,<3`

Note: `richdem` does not provide precompiled wheels through PyPI, so it is installed from conda-forge, thus to use `aywen` it is recommended to use `pixi` or `conda` for environment management and dependency resolution.

## Installation

### Using pixi (Development & Building)
This project uses [pixi](https://pixi.sh) for dependency management, providing better cross-platform compatibility and handling of scientific packages.

**Install pixi first:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
# or on Windows with PowerShell:
# iwr -useb https://pixi.sh/install.ps1 | iex
```

**Then install the project:**
```bash
# Clone repo and go to development branch
git clone https://github.com/RNS-ago/aywen.git
cd aywen

# Install all dependencies and the package
pixi install

# Activate the environment
pixi shell
```

**Building package from source:**
```bash
# Clone repo and go to directory
git clone https://github.com/RNS-ago/aywen.git
cd aywen

# Build packages
pixi run build-conda    # Build conda package

```

### Using conda (From .conda)
Once you have built the package, you can install it using conda:

**macOS/Linux (bash):**
```bash
# Create and activate conda environment
conda create -n aywen python=3.11
conda activate aywen

# Install package from local directory
conda install -c file:///path/to/aywen/dist/conda/aywen-x.x-xxxxxx.conda
```


**Windows (PowerShell):**
```powershell
# Create and activate conda environment
conda create -n aywen python=3.11
conda activate aywen

# Install package from local directory
conda install -c file:///path/to/aywen/dist/conda/aywen-x.x-xxxxxx.conda

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
from aywen.features import add_zones_to_df, split_zones, get_zones

# Get a dictionary of zones for the coordinates
zones = get_zones(
    coords = (-33.4489, -70.6693),
    shapefile_path="data/zones.shp",
    crs: str = "EPSG:4326",
    lon_col="longitude",
    lat_col="latitude",
    zone_col="area",
    new_zone_col="zone_alert",
    )


# Add zone labels to fire incidents
gdf = add_zones_to_df(
    df=fire_df,
    shapefile_path="data/zones.shp",
    crs: str = "EPSG:4326",
    lon_col="longitude",
    lat_col="latitude",
    zone_col="area",
    new_zone_col="zone_alert",
)

splited_gdf = split_zones(
    df = gdf,
    zone_col = "zone_alert",
    separation_keys = ["zone_WE", "zone_NS"]
    )

```

### Logging functionality
Logging is integrated throughout the pipeline to provide insights into the processing steps. The logging level can be adjusted in the configuration.

To enable the built in logging functionality run the following commands before you run any function from the library.
```python
import logging
from aywen.logging_setup import configure_logging

# Set up logging to aywens logging config
configure_logging()

# Set the logging level for the aywen_logger, this case DEBUG
logging.getLogger("aywen_logger").setLevel(logging.DEBUG)
```

## Development

### Environment Management
This project uses pixi for dependency management with a hybrid configuration that supports both conda and PyPI packages:

```bash
# Set up development environment
pixi install

# Activate environment
pixi shell

# Run tests
pixi run python -m pytest  # if you have tests

# Build packages
pixi run build-conda    # Build conda package
pixi run build-wheel    # Build PyPI wheel
pixi run build-all      # Build all package types
```

### For Contributors
- **Preferred**: Use pixi for the best experience with scientific dependencies
- **Alternative**: Standard pip workflow is also supported via `pyproject.toml`
- Dependencies are locked in `pixi.lock` for reproducible environments
- The project builds both conda packages (via pixi-build) and PyPI wheels

### Package Structure
- Configuration: `pyproject.toml` (hybrid pixi + standard Python)
- Lock file: `pixi.lock`
- Dependencies: Conda packages preferred, with PyPI fallback for unavailable packages

## Notes
- Expected columns referenced by helpers (e.g., `hr_arribo`, `glosa`, `recurso`, `fire_id`, `arrival_datetime_inc`) are standardized inside the pipeline; see `src/aywen/preprocessing.py` and `src/aywen/features.py` for details.
- Logging is preconfigured with colorized console output; see `src/aywen/logging_setup.py`.
- For best compatibility with geospatial dependencies (GDAL, GEOS, etc.), pixi is recommended over pip.

## Contributing
Contributions welcome! Please:
1. Use pixi for development setup (required due to richdem dependency)
2. Open an issue or PR with a clear description
3. Include minimal reproducible examples
4. Follow existing code style and logging patterns