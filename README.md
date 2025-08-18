# 🏀 NBA Era Comparison Machine

## Project Overview
A machine learning and data analysis web app that compares offensive style and efficiency across NBA eras (1990s, 2000s, 2010s, 2020s). Built for casual fans to explore NBA offensive trends while demonstrating technical skills for professional portfolios.

## 🎯 Project Objectives
- Analyze how NBA offensive strategy has evolved over time
- Build a classifier that predicts which era a team belongs to based on offensive stats
- Cluster teams into offensive playstyle types regardless of era
- Visualize all results in a clean, interactive dashboard using Streamlit

## 🧩 Core Features

### 1. Descriptive Era Analysis
Compare metrics like: PTS, FGA, 3PA, 3P%, AST, ORtg, TS%, Pace, eFG%, TOV, AST%, USG%

Show trends over time using:
- Line graphs (e.g., 3PA growth)
- Bar charts (stat comparison by era)
- Radar charts (positional archetypes, optional)

### 2. ML Predictive Model – "Guess the Era"
- **Supervised learning**: Random Forest Classifier
- **Input**: Offensive stat line (user input or real data)
- **Output**: Predicted era + class probabilities
- **Additional**: Feature importance chart, confusion matrix

### 3. Clustering – "Offensive Archetypes"
- Use KMeans or Agglomerative Clustering
- Cluster teams using stats like: 3PA, PTS, AST, ORtg, Pace, etc.
- **Output**: Label clusters (e.g., "Pace & Space", "Iso Era", "Ball-Movement Era")
- Show 2D visual (scatter plot) of teams by cluster

### 4. Interactive Dashboard
Built with Streamlit using multi-tab layout:
- 🏠 Era Overview
- 📊 Compare Eras
- 🤖 Predict the Era
- 🧠 Offensive Archetypes
- 📁 About

## 🛠️ Tools & Libraries
- Python, Pandas, Scikit-learn
- Plotly or Matplotlib/Seaborn
- Streamlit for dashboard
- joblib for model saving
- Basketball Reference or NBA API for data

## 📁 Project Structure
```
nba-era-comparison/
│
├── data/                     # Raw + processed data
│   └── raw_stats.csv         # NBA team offensive stats
│
├── models/                   # Saved ML model files
│   └── era_classifier.pkl    # Trained Random Forest model
│
├── app.py                    # Streamlit main dashboard
├── ml_predictor.py           # Era prediction functions
├── cluster_analysis.py       # Offensive archetype clustering
├── data_loader.py            # Data loading / cleaning logic
├── utils.py                  # Helper functions
└── README.md                 # Project overview
```

## 🚀 Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Add NBA data to `data/raw_stats.csv`
3. Run the app: `streamlit run app.py`

## 📊 Key Metrics Analyzed
- **PTS**: Points per game
- **FGA**: Field goal attempts
- **3PA**: Three-point attempts
- **3P%**: Three-point percentage
- **AST**: Assists
- **ORtg**: Offensive rating
- **TS%**: True shooting percentage
- **Pace**: Possessions per game
- **eFG%**: Effective field goal percentage
- **TOV**: Turnovers
- **AST%**: Assist percentage
- **USG%**: Usage rate
