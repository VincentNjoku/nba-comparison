"""
Simple NBA Era Predictor
Uses Random Forest to predict which NBA era a team belongs to.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os

def train_model():
    """
    Train and save the era prediction model
    """
    print("🤖 Training NBA Era Classifier...")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Load data
    df = pd.read_csv("data/cleaned_stats.csv")
    
    # Features and target
    features = ['PTS', 'FGA', '3PA', '3P%', 'AST', 'TOV', '3P_Rate', 'Pace', 'ORtg']
    X = df[features]
    y = df['Era']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    cv_score = cross_val_score(model, X, y, cv=5).mean()
    
    print(f"✅ Model trained!")
    print(f"📊 Accuracy: {accuracy:.1%}")
    print(f"📊 Cross-validation: {cv_score:.1%}")
    
    # Save model
    joblib.dump(model, "models/era_classifier.pkl")
    print("💾 Model saved!")
    
    return model, accuracy

def predict_era(team_stats):
    """
    Predict era for a team's stats
    """
    # Load model
    model = joblib.load("models/era_classifier.pkl")
    
    # Required features
    features = ['PTS', 'FGA', '3PA', '3P%', 'AST', 'TOV', '3P_Rate', 'Pace', 'ORtg']
    
    # Check features
    if not all(f in team_stats for f in features):
        return "❌ Missing stats"
    
    # Make prediction using DataFrame to preserve feature names
    X = pd.DataFrame([team_stats], columns=features)
    era = model.predict(X)[0]
    confidence = float(max(model.predict_proba(X)[0]))
    
    return f"{era} ({confidence:.1%} confidence)"

if __name__ == "__main__":
    # Train model
    model, accuracy = train_model()
    
    # Test with sample data
    print("\n🧪 Testing Predictions:")
    
    sample_teams = {
        '1990s Bulls': {'PTS': 105, 'FGA': 87, '3PA': 12, '3P%': 0.32, 'AST': 26, 'TOV': 15, '3P_Rate': 0.14, 'Pace': 96, 'ORtg': 1.10},
        '2020s Bucks': {'PTS': 118, 'FGA': 89, '3PA': 35, '3P%': 0.37, 'AST': 26, 'TOV': 13, '3P_Rate': 0.39, 'Pace': 99, 'ORtg': 1.20}
    }
    
    for team, stats in sample_teams.items():
        prediction = predict_era(stats)
        print(f"🏀 {team}: {prediction}")
    
    print(f"\n✅ Done! Model accuracy: {accuracy:.1%}")
