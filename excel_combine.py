from math import e
import pandas as pd

# Path to your multi-sheet Excel file
excel_path = "data/nba_team_stats_1990_to_2025.xlsx"

# Load all sheets
xls = pd.ExcelFile(excel_path, engine='openpyxl')
all_dfs = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df['Season'] = int(sheet_name)  # Add season as column
    all_dfs.append(df)

# Combine sheets
combined_df = pd.concat(all_dfs, ignore_index=True)

# Clean repeated headers or league averages
combined_df = combined_df[combined_df['Team'] != 'League Average']
combined_df = combined_df[combined_df['Team'] != 'Team']

# Save to CSV for ML
combined_df.to_csv("data/raw_stats.csv", index=False)

# Save to Excel for visual inspection
combined_df.to_excel("data/raw_stats_combined.xlsx", index=False)

print("✅ Data saved to both raw_stats.csv and raw_stats_combined.xlsx")
