import pytest
from aywen.utils import load_latest_run
import logging
from aywen.logging_setup import configure_logging
from aywen.utils import load_latest_run
from aywen.testing import assert_df

# ---- HARD-CODED VALUES (edit these) ----
EXPERIMENT1 = "arauco_fire_pipeline"
RUN_NAME1   = "kitral-run"

EXPERIMENT2 = "arauco_fire_evaluation"
RUN_NAME2   = "kitral-run"
# ----------------------------------------

logger = logging.getLogger("aywen_logger")
logger.setLevel(logging.INFO)


def test_loaded_dataframes_on_common_columns_latest_runs():
    loaded1 = load_latest_run(EXPERIMENT1, run_name=RUN_NAME1)
    loaded2 = load_latest_run(EXPERIMENT2, run_name=RUN_NAME2)

    df1 = loaded1["df"]
    df2 = loaded2["df"]


    assert df1 is not None, f"No dataframe artifact in latest run for experiment {EXPERIMENT1!r}"
    assert df2 is not None, f"No dataframe artifact in latest run for experiment {EXPERIMENT2!r}"

    assert_df(df1, df2, time_cols=["start_datetime"])


