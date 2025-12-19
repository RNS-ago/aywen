# conftest.py
import os
from pathlib import Path
import pytest
import mlflow

@pytest.fixture(autouse=True, scope="session")
def set_mlflow_tracking_uri():
    store_uri = Path(r"C:\Users\lopez\GitRepos\aywen\scripts\mlruns").resolve().as_uri()
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(store_uri)
    os.environ["MLFLOW_TRACKING_URI"] = store_uri  # keep both in sync
    yield
    # restore (optional)
    mlflow.set_tracking_uri(prev if prev else "")

