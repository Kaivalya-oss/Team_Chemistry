import json
import pandas as pd
# Load the data generated in Step 4
df_teams = pd.read_csv("teams_with_chemistry_index.csv")

# Select only necessary columns to keep the file lightweight
required_cols = [
    'Club', 'SquadSize', 'NationalityDiversity', 'Attack_Work_Intensity', 
    'Defense_Work_Intensity', 'Age', 'International Reputation', 'Chemistry_Index_100'
]

# Convert to a dictionary with Club as the key
team_data_dict = df_teams[required_cols].set_index('Club').to_dict(orient='index')

# Save as JSON
with open('teams_data.json', 'w') as f:
    json.dump(team_data_dict, f, indent=4)

print("Exported teams_data.json for frontend use.")