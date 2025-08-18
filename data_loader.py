"""
Simple NBA Data Loader
Loads, cleans, and prepares NBA team stats for analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_data():
    """
    Main function: Load raw data, clean it, add era labels, and save
    """
    print("🏀 Loading NBA team stats...")
    
    # Load raw data
    df = pd.read_csv("data/raw_stats.csv")
    print(f"✅ Loaded {len(df)} records")
    
    # Clean the data
    df = clean_data(df)
    
    # Save cleaned data
    df.to_csv("data/cleaned_stats.csv", index=False)
    print(f"✅ Saved cleaned data: {len(df)} records")
    
    return df

def clean_data(df):
    """
    Clean and prepare the data
    """
    print("🧹 Cleaning data...")
    
    # Remove unnamed columns (they're empty anyway)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"   Removed {len(unnamed_cols)} empty columns")
    
    # Handle missing values - only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print(f"   Filled missing values in {len(numeric_cols)} numeric columns")
    
    # Standardize team names
    df = standardize_teams(df)
    
    # Add year column
    df['Year'] = df['Season'].astype(int)
    
    # Add era labels
    df['Era'] = df['Year'].apply(get_era)
    
    # Add key derived metrics
    df['3P_Rate'] = df['3PA'] / df['FGA']  # What % of shots are 3s
    df['Pace'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']  # Approximate possessions
    df['ORtg'] = df['PTS'] / df['Pace']  # Points per possession
    
    # Select final columns
    final_cols = ['Year', 'Team', 'PTS', 'FGA', '3PA', '3P%', 'AST', 'TOV', 
                  '3P_Rate', 'Pace', 'ORtg', 'Era']
    df = df[final_cols]
    
    print(f"   Final shape: {df.shape}")
    return df

def standardize_teams(df):
    """
    Convert team names to abbreviations
    """
    # Normalize team strings (remove asterisks from playoff teams, trim spaces)
    if 'Team' in df.columns:
        df['Team'] = df['Team'].astype(str).str.replace('*', '', regex=False).str.strip()
    
    team_map = {
        'Golden State Warriors': 'GSW',
        'Phoenix Suns': 'PHX', 
        'Denver Nuggets': 'DEN',
        'Portland Trail Blazers': 'POR',
        'Orlando Magic': 'ORL',
        'Los Angeles Lakers': 'LAL',
        'Boston Celtics': 'BOS',
        'Chicago Bulls': 'CHI',
        'Miami Heat': 'MIA',
        'Houston Rockets': 'HOU',
        'Washington Wizards': 'WAS',
        'Washington Bullets': 'WAS',
        'New York Knicks': 'NYK',
        'Atlanta Hawks': 'ATL',
        'Detroit Pistons': 'DET',
        'Cleveland Cavaliers': 'CLE',
        'Indiana Pacers': 'IND',
        'Milwaukee Bucks': 'MIL',
        'Toronto Raptors': 'TOR',
        'Philadelphia 76ers': 'PHI',
        'Brooklyn Nets': 'BKN',
        'New Jersey Nets': 'BKN',
        'Charlotte Hornets': 'CHA',
        'Charlotte Bobcats': 'CHA',
        'Minnesota Timberwolves': 'MIN',
        'Sacramento Kings': 'SAC',
        'Utah Jazz': 'UTA',
        'Dallas Mavericks': 'DAL',
        'San Antonio Spurs': 'SAS',
        'Memphis Grizzlies': 'MEM',
        'Vancouver Grizzlies': 'MEM',
        'New Orleans Pelicans': 'NOP',
        'New Orleans Hornets': 'NOP',
        'Oklahoma City Thunder': 'OKC',
        'Seattle SuperSonics': 'OKC',
        'Los Angeles Clippers': 'LAC',
        'Sacramento Kings': 'SAC'
    }
    
    df['Team'] = df['Team'].map(team_map).fillna(df['Team'])
    return df

def get_era(year):
    """
    Convert year to era label
    """
    if year < 2000:
        return '1990s'
    elif year < 2010:
        return '2000s'
    elif year < 2020:
        return '2010s'
    else:
        return '2020s'

def prepare_ml_data(df):
    """
    Prepare data for machine learning
    """
    # Features for ML (excluding Year, Team, Era)
    features = ['PTS', 'FGA', '3PA', '3P%', 'AST', 'TOV', '3P_Rate', 'Pace', 'ORtg']
    
    X = df[features]
    y = df['Era']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🤖 ML data prepared:")
    print(f"   Features: {len(features)}")
    print(f"   Train: {len(X_train)} records")
    print(f"   Test: {len(X_test)} records")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load and clean data
    df = load_and_clean_data()
    
    # Show summary
    print(f"\n📊 Data Summary:")
    print(f"   Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"   Teams: {df['Team'].nunique()}")
    print(f"   Eras: {df['Era'].value_counts().to_dict()}")
    
    # Prepare ML data
    X_train, X_test, y_train, y_test = prepare_ml_data(df)
    
    print("\n✅ Data loader complete!")
