# add_to_team.py
import pandas as pd
import numpy as np
import pickle

# Load your best model (from step 8)
with open('best_gb_model.pkl', 'rb') as f:          # change filename if different
    model = pickle.load(f)

def simulate_player_swap(team_name: str, remove_player_name: str, add_player_name: str) -> dict:
    players = pd.read_csv("players_for_dashboard.csv")
    
    # 1. Get current team
    current_team = players[players['Club'] == team_name].copy()
    if current_team.empty:
        return {"error": f"Team {team_name} not found"}
    
    # 2. Validate removal
    if remove_player_name not in current_team['Name'].values:
        return {"error": f"{remove_player_name} is not in {team_name}"}
        
    # 3. Validate addition
    # (Optional: check if already exists, though UI filters it usually)
    if add_player_name in current_team['Name'].values:
        return {"error": f"{add_player_name} is already in {team_name}"}
    
    new_player_row = players[players['Name'] == add_player_name]
    if new_player_row.empty:
        return {"error": f"Player {add_player_name} not found"}
    new_player_row = new_player_row.iloc[0]

    # --- BEFORE STATE (Current Team) ---
    curr_n = len(current_team)
    curr_age = current_team['Age'].mean()
    curr_attack = current_team['Attack_Work_Intensity'].mean()
    curr_defense = current_team['Defense_Work_Intensity'].mean()
    curr_rep = current_team['International Reputation'].mean()
    curr_diversity = current_team['Nationality'].nunique()
    curr_balance = 1 / (1 + abs(curr_attack - curr_defense))
    curr_age_close = 1 - abs(curr_age - 26.5) / current_team['Age'].std() if curr_n > 1 else 0.5

    # --- AFTER STATE (Swap) ---
    # Remove old player
    temp_team = current_team[current_team['Name'] != remove_player_name]
    # Add new player
    new_team = pd.concat([temp_team, new_player_row.to_frame().T], ignore_index=True)
    
    new_n = len(new_team)
    new_age = new_team['Age'].mean()
    new_attack = new_team['Attack_Work_Intensity'].mean()
    new_defense = new_team['Defense_Work_Intensity'].mean()
    new_rep = new_team['International Reputation'].mean()
    new_diversity = new_team['Nationality'].nunique()
    new_balance = 1 / (1 + abs(new_attack - new_defense))
    new_age_close = 1 - abs(new_age - 26.5) / new_team['Age'].std() if new_n > 1 else 0.5
    
    # Normalize (MUST match how you trained the model!)
    norm = lambda v, mn, mx: (v - mn) / (mx - mn) if mx > mn else 0.5
    
    mins = {'div': -1.7074, 'bal': 0.2101, 'rep': -0.4920, 'agec': -28.6977}
    maxs = {'div': 3.2591, 'bal': 0.9967, 'rep': 6.4640, 'agec': -21.2153}
    
    before_vec = np.array([
        norm(curr_diversity, mins['div'], maxs['div']),
        norm(curr_balance, mins['bal'], maxs['bal']),
        norm(curr_rep, mins['rep'], maxs['rep']),
        norm(curr_age_close, mins['agec'], maxs['agec'])
    ]).reshape(1, -1)
    
    after_vec = np.array([
        norm(new_diversity, mins['div'], maxs['div']),
        norm(new_balance, mins['bal'], maxs['bal']),
        norm(new_rep, mins['rep'], maxs['rep']),
        norm(new_age_close, mins['agec'], maxs['agec'])
    ]).reshape(1, -1)
    
    before_score = model.predict(before_vec)[0]
    after_score  = model.predict(after_vec)[0]
    
    return {
        "team": team_name,
        "removed": remove_player_name,
        "added": add_player_name,
        "before_chemistry": round(float(before_score), 2),
        "after_chemistry": round(float(after_score), 2),
        "change": round(float(after_score - before_score), 2),
        "metrics": {
            "labels": ["Diversity", "Balance", "Reputation", "Age Structure"],
            "before": [round(float(x), 2) for x in before_vec[0]],
            "after": [round(float(x), 2) for x in after_vec[0]]
        }
    }
# Test
# print(simulate_player_addition("FC Barcelona", "Lionel Messi"))