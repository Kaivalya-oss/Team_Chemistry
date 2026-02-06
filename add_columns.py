import pandas as pd
import numpy as np

# Load original data to get Work Rates
df_orig = pd.read_excel("fifa_eda_stats.xlsx")

# Work Rate parsing logic from step2.py
workrate_map = {"Low": 1, "Medium": 2, "High": 3}
def parse_work_rate(wr):
    if pd.isna(wr):
        return 2.0, 2.0
    wr_clean = str(wr).strip().replace(" / ", "/").replace("  ", " ")
    parts = [p.strip() for p in wr_clean.split("/")]
    if len(parts) != 2:
        return 2.0, 2.0
    att = workrate_map.get(parts[0], 2)
    def_ = workrate_map.get(parts[1], 2)
    return att, def_

print("Parsing Work Rates...")
df_orig['parsed_wr'] = df_orig["Work Rate"].apply(parse_work_rate)
df_orig["Attack_Work_Intensity"] = df_orig['parsed_wr'].apply(lambda x: x[0])
df_orig["Defense_Work_Intensity"] = df_orig['parsed_wr'].apply(lambda x: x[1])

# Load current dashboard players
df_dash = pd.read_csv("players_for_dashboard.csv")
print(f"Original dashboard columns: {df_dash.columns.tolist()}")

# Merge based on Name and Club (to avoid duplicates issues)
# Note: dashboard file might have duplicates if names are same, but Name+Club should be unique mostly
cols_to_merge = ["Name", "Club", "Attack_Work_Intensity", "Defense_Work_Intensity"]
merged_df = pd.merge(df_dash, df_orig[cols_to_merge], on=["Name", "Club"], how="left")

# Fill missing (if any mismatch) with default medium (2.0)
merged_df["Attack_Work_Intensity"] = merged_df["Attack_Work_Intensity"].fillna(2.0)
merged_df["Defense_Work_Intensity"] = merged_df["Defense_Work_Intensity"].fillna(2.0)

print(f"New dashboard columns: {merged_df.columns.tolist()}")

# Overwrite the file
merged_df.to_csv("players_for_dashboard.csv", index=False)
print("Updated players_for_dashboard.csv with Work Intensity columns.")
