from functools import wraps
import datetime as dt
import pandas as pd
import logging
import os

# --- Configure logging globally ---
logging.basicConfig(
    level=logging.INFO,  # Default level (can be changed later)
    format="%(message)s"  # Only show the message (no timestamps/levels unless you want them)
)

# --- Decorator ---
def log_step(func, key="fire_id"):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)

        if isinstance(result, pd.DataFrame):
            logging.info(
                f"\033[94m[LOG]\033[0m Just ran step "
                f"\033[92m{func.__name__}\033[0m, "
                f"there are \033[93m{len(result[key].unique())}\033[0m unique {key} values, "
                f"and it took \033[91m{time_taken}\033[0m"
            )

        elif isinstance(result, list) and all(isinstance(df, pd.DataFrame) for df in result):
            logging.info(
                f"\033[94m[LOG]\033[0m Just ran step "
                f"\033[92m{func.__name__}\033[0m, "
                f"it returned a list of {len(result)} DataFrames "
                f"and took \033[91m{time_taken}\033[0m"
            )
            for i, df in enumerate(result, 1):
                df_name = getattr(df, "name", f"df_{i}")
                logging.info(
                    f"\t\033[93mDataset {df_name}\033[0m: "
                    f"{df.shape[0]} rows x {df.shape[1]} columns"
                )

        else:
            logging.info(
                f"\033[94m[LOG]\033[0m Just ran step "
                f"\033[92m{func.__name__}\033[0m, "
                f"it took \033[91m{time_taken}\033[0m"
            )
        return result
    return wrapper


def concatnate_name_from_paths(paths: list[str]) -> str:
    report_type = os.path.splitext(os.path.basename(paths[0]))[0].split('_')[0]
    
    report_date_range = []
    for path in paths:
        report_date_range.append(os.path.splitext(os.path.basename(path))[0].split('_')[1])
        
    report_date_range = '_'.join(report_date_range)
    
    return f"{report_type}_{report_date_range}"
    