import pandas as pd
from aywen.evaluating import INPUT_VARIABLES

# ---------------- paths  ----------------
DIR_ROOT =  "G:/Shared drives/OpturionHome/AraucoFire"
DATA_PROCESSED = DIR_ROOT + "/2_data/processed/in_mlruns"

# Load Parquet
df = pd.read_parquet(DATA_PROCESSED + "/data.parquet", columns=INPUT_VARIABLES)
print("Loaded data.parquet with shape:", df.shape)
print("Loaded data.parquet with columns:", df.columns.tolist())

# Save as CSV
df.to_csv(DATA_PROCESSED + "/data.csv", index=False)
print("Converted data.parquet to data.csv")