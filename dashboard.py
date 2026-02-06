import pandas as pd

# 1. Load player data
try:
    players = pd.read_excel("fifa_eda_stats.xlsx")
except FileNotFoundError:
    players = pd.read_csv("cleaned_players.csv")

# Clean column names (removes hidden spaces)
players.columns = players.columns.str.strip()

# Define the columns you WANT vs. what ACTUALLY exists
desired_p_cols = ['Name', 'Club', 'Age', 'Attack_Work_Intensity', 'Defense_Work_Intensity', 
                  'International Reputation', 'Nationality']

# Filter to only include columns found in the file
existing_p_cols = [col for col in desired_p_cols if col in players.columns]
players = players[existing_p_cols].dropna(subset=['Club'])

players.to_csv("players_for_dashboard.csv", index=False)
print(f"Saved players_for_dashboard.csv → {len(players)} players")
print(f"Columns used: {existing_p_cols}")

# 2. Load teams file
teams = pd.read_csv("teams_with_chemistry_index.csv")
teams.columns = teams.columns.str.strip()

desired_t_cols = ['Club', 'Chemistry_Index_100', 'NationalityDiversity', 'WorkRate_Balance',
                  'International Reputation', 'Age']

existing_t_cols = [col for col in desired_t_cols if col in teams.columns]
teams = teams[existing_t_cols]

teams.to_csv("teams_for_dashboard.csv", index=False)
print(f"Saved teams_for_dashboard.csv → {len(teams)} teams")