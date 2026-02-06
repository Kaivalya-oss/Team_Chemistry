# ============================================================
# STEP 2: DATA PREPROCESSING
# Project: Team Chemistry in Football
# ============================================================

# -----------------------------
# Import libraries (add to step 1 imports if not already there)
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# 1. Load the data (assuming from step 1)
# -----------------------------
df = pd.read_excel("fifa_eda_stats.xlsx")

# Quick safety copy
df_clean = df.copy()

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
# Critical columns we care about
print("Missing values before cleaning:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# Loaned From → fill with 'Not Loaned'
df_clean["Loaned From"] = df_clean["Loaned From"].fillna("Not Loaned")

# Joined → can drop or fill with placeholder (we'll mostly ignore for team agg)
df_clean["Joined"] = df_clean["Joined"].fillna(0)

# Contract Valid Until → same
df_clean["Contract Valid Until"] = df_clean["Contract Valid Until"].fillna(0)

# Club → very few missing → drop those rows (usually free agents)
df_clean = df_clean.dropna(subset=["Club"])

# Body Type, Position, etc. → fill with mode or 'Unknown'
for col in ["Body Type", "Position", "Work Rate"]:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Remaining small missing → median for numerics
numeric_cols = df_clean.select_dtypes(include=np.number).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

print("\nMissing values after basic cleaning:", df_clean.isnull().sum().sum())

# -----------------------------
# 3. Clean & Convert Money Columns (Value, Wage, Release Clause)
# -----------------------------
def money_to_numeric(x):
    if pd.isna(x) or x == 0:
        return 0.0
    x = str(x).replace("€", "").strip()
    if "M" in x:
        return float(x.replace("M", "")) * 1_000_000
    elif "K" in x:
        return float(x.replace("K", "")) * 1_000
    else:
        return float(x)

for col in ["Value", "Wage", "Release Clause"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(money_to_numeric)

print("\nSample after money conversion:")
print(df_clean[["Name", "Club", "Value", "Wage", "Release Clause"]].head())

# -----------------------------
# 4. Feature Engineering (before aggregation)
# -----------------------------
# Preferred Foot → binary
df_clean["is_LeftFoot"] = (df_clean["Preferred Foot"] == "Left").astype(int)

# Safer Work Rate handling
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

df_clean[["Attack_Work_Intensity", "Defense_Work_Intensity"]] = \
    df_clean["Work Rate"].apply(parse_work_rate).tolist()

# Optional: difference or compatibility score later
df_clean["WorkRate_Balance"] = abs(df_clean["Attack_Work_Intensity"] - df_clean["Defense_Work_Intensity"])

# International Reputation, Skill Moves, Weak Foot → already ordinal, good

# -----------------------------
# 5. Aggregation to TEAM level (core of team chemistry proxy prep)
# -----------------------------
team_group = df_clean.groupby("Club")

# Define aggregation dictionary
agg_dict = {
    # Count players per team (squad size)
    "Name": "count",
    
    # Average age & experience-related
    "Age": "mean",
    "Overall": "mean",
    "Potential": "mean",
    
    # Financial strength
    "Value": "mean",
    "Wage": "mean",
    
    # Technical / physical averages
    "Crossing": "mean",
    "Finishing": "mean",
    "ShortPassing": "mean",
    "LongPassing": "mean",
    "Dribbling": "mean",
    "BallControl": "mean",
    "Acceleration": "mean",
    "SprintSpeed": "mean",
    "Stamina": "mean",
    "Strength": "mean",
    "Aggression": "mean",
    "Interceptions": "mean",
    "Positioning": "mean",
    "Vision": "mean",
    "Composure": "mean",
    
    # Defensive
    "Marking": "mean",
    "StandingTackle": "mean",
    "SlidingTackle": "mean",
    
    # Goalkeeping (will be low for non-GKs → can separate later)
    "GKDiving": "mean",
    
    # Chemistry-related proxies
    "is_LeftFoot": "mean",                     # % left-footed
    "International Reputation": "mean",
    "Skill Moves": "mean",
    "Weak Foot": "mean",
    "Attack_Work_Intensity": "mean",
    "Defense_Work_Intensity": "mean",
    
    # Diversity proxies (you can improve later)
    "Nationality": lambda x: x.nunique(),       # number of nationalities
}

# Apply aggregation
team_df = team_group.agg(agg_dict).reset_index()

# Rename columns for clarity
team_df = team_df.rename(columns={
    "Name": "SquadSize",
    "Nationality": "NationalityDiversity"
})

print("\nTeam-level dataset preview:")
print(team_df.head())

print("\nTeam dataset shape:", team_df.shape)

# -----------------------------
# 6. Scaling numerical features (team level)
# -----------------------------
numeric_team_cols = team_df.select_dtypes(include=np.number).columns.drop("SquadSize", errors="ignore")  # don't scale count

scaler = StandardScaler()
team_df[numeric_team_cols] = scaler.fit_transform(team_df[numeric_team_cols])

print("\nScaled team data sample:")
print(team_df.head())

# -----------------------------
# 7. Save processed data for next steps
# -----------------------------
team_df.to_csv("team_level_processed.csv", index=False)
print("\nSaved team-level processed data → ready for EDA & target construction")