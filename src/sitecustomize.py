# sitecustomize.py — auto-run on Python startup if on sys.path
from pathlib import Path
import sys, os
try:
    from pyproj import datadir
except Exception:
    # pyproj not installed; nothing to do
    datadir = None

proj_dir = Path(sys.prefix) / "Library" / "share" / "proj"
if proj_dir.is_dir() and (proj_dir / "proj.db").exists():
    os.environ.setdefault("PROJ_DATA", str(proj_dir))
    os.environ.setdefault("PROJ_LIB",  str(proj_dir))
    if datadir is not None:
        datadir.set_data_dir(str(proj_dir))
