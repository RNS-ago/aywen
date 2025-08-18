from aywen import preprocessing as pp
import pandas as pd
from aywen.logging_setup import configure_logging
import logging
configure_logging()
logging.getLogger("aywen_logger").setLevel(logging.INFO)

raw_data_dir = "testing/test_raw_data"
preprocessd_output_dir = "testing/test_preprocessed_data"

fire_data_paths = [f"{raw_data_dir}/Incendios_2014-2018.csv", f"{raw_data_dir}/Incendios_2021-2025.csv"]
dispatch_data_paths = [f"{raw_data_dir}/Despachos_2014-2018.csv", f"{raw_data_dir}/Despachos_2021-2025.csv"]

pp.preprocess_pipeline(fire_data_paths=fire_data_paths, dispatch_data_paths=dispatch_data_paths, output_dir=preprocessd_output_dir)