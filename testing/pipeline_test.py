import aywen
from aywen import preprocessing as pp
import pandas as pd
from aywen.logging_setup import configure_logging
import logging
from pathlib import Path
configure_logging()
logging.getLogger("aywen_logger").setLevel(logging.INFO)

# Find the base directory of the library


base_library_dir = Path(aywen.__path__[0]).parent.parent
print(f"Base library directory: {base_library_dir}")

# Construct paths using Path objects
raw_data_dir = base_library_dir / "testing" / "test_raw_data"
preprocessed_output_dir = base_library_dir / "testing" / "test_preprocessed_data"

fire_data_paths = [
    raw_data_dir / "Incendios_2014-2018.csv",
    raw_data_dir / "Incendios_2021-2025.csv"
]

dispatch_data_paths = [
    raw_data_dir / "Despachos_2014-2018.csv", 
    raw_data_dir / "Despachos_2021-2025.csv"
]

# Convert to strings if your function expects strings
pp.preprocess_pipeline(
    fire_data_paths=[str(p) for p in fire_data_paths], 
    dispatch_data_paths=[str(p) for p in dispatch_data_paths], 
    output_dir=str(preprocessed_output_dir)
)