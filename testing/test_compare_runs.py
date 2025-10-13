import pytest
from pandas.testing import assert_frame_equal

# adjust import paths
from aywen.utils import load_latest_run

# ---- HARD-CODED VALUES (edit these) ----
EXPERIMENT1 = "arauco_fire_pipeline"
RUN_NAME1   = "kitral-run"

EXPERIMENT2 = "arauco_fire_evaluation"
RUN_NAME2   = "kitral-run"
# ----------------------------------------


def test_loaded_dataframes_on_common_columns_latest_runs():
    loaded1 = load_latest_run(EXPERIMENT1, run_name=RUN_NAME1)
    loaded2 = load_latest_run(EXPERIMENT2, run_name=RUN_NAME2)

    df1 = loaded1["df"]
    df2 = loaded2["df"]

    assert df1 is not None, f"No dataframe artifact in latest run for experiment {EXPERIMENT1!r}"
    assert df2 is not None, f"No dataframe artifact in latest run for experiment {EXPERIMENT2!r}"

    # Compare ONLY on common columns
    common_cols = df1.columns.intersection(df2.columns)
    assert len(common_cols) > 0, "No common columns between the two dataframes."

    common_cols = list(common_cols)
    df1c = df1[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    df2c = df2[common_cols].sort_values(by=common_cols).reset_index(drop=True)

    assert_frame_equal(df1c, df2c, check_exact=True)

