import pandas as pd
df = pd.read_csv("players_for_dashboard.csv")
print(f"Total Rows: {len(df)}")
print(f"Unique Names: {df['Name'].nunique()}")
