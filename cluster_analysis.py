"""
NBA Offensive Archetypes - Clustering Analysis
Groups teams by offensive playing style regardless of era.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def create_offensive_archetypes():
    """
    Create offensive archetypes using K-means clustering
    """
    print("🧠 Creating NBA Offensive Archetypes...")
    
    # Load cleaned data
    df = pd.read_csv("data/cleaned_stats.csv")
    
    # Select features for clustering (offensive style indicators)
    features = ['PTS', '3PA', '3P_Rate', 'AST', 'Pace', 'ORtg']
    X = df[features]
    
    print(f"📊 Using {len(features)} features: {features}")
    print(f"🏀 Analyzing {len(df)} team seasons")
    
    # Scale the features (important for clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters using elbow method
    inertias = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Choose 4 clusters (representing main offensive styles)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters and assign meaningful labels
    cluster_labels = analyze_clusters(df, features)
    df['Archetype'] = df['Cluster'].map(cluster_labels)
    
    print(f"✅ Created {n_clusters} offensive archetypes!")
    
    # Show cluster summary
    show_cluster_summary(df)
    
    # Save results
    df.to_csv("data/teams_with_archetypes.csv", index=False)
    print("💾 Results saved to: data/teams_with_archetypes.csv")
    
    return df, kmeans, scaler

def analyze_clusters(df, features):
    """
    Analyze clusters and assign meaningful labels
    """
    cluster_means = df.groupby('Cluster')[features].mean()
    
    print(f"\n🔍 Cluster Analysis:")
    for cluster_id in range(len(cluster_means)):
        print(f"   Cluster {cluster_id}: {dict(cluster_means.loc[cluster_id])}")
    
    # Create archetype labels based on cluster characteristics
    archetypes = {}
    
    for cluster_id in range(len(cluster_means)):
        cluster_data = cluster_means.loc[cluster_id]
        
        # Based on the cluster analysis we can see:
        if cluster_data['3P_Rate'] > 0.35:  # Very high 3-point rate
            archetype = "Modern 3-Point"
        elif cluster_data['3P_Rate'] > 0.25:  # High 3-point rate
            archetype = "Pace & Space"
        elif cluster_data['3P_Rate'] < 0.12:  # Very low 3-point rate
            archetype = "Traditional"
        else:  # Moderate 3-point rate
            archetype = "Balanced"
        
        archetypes[cluster_id] = archetype
    
    return archetypes

def show_cluster_summary(df):
    """
    Display summary of each offensive archetype
    """
    print("\n📊 Offensive Archetypes Summary:")
    print("=" * 50)
    
    for archetype in df['Archetype'].unique():
        cluster_data = df[df['Archetype'] == archetype]
        
        print(f"\n🏀 {archetype}:")
        print(f"   Teams: {len(cluster_data)}")
        print(f"   Years: {cluster_data['Year'].min()} - {cluster_data['Year'].max()}")
        print(f"   Avg PTS: {cluster_data['PTS'].mean():.1f}")
        print(f"   Avg 3PA: {cluster_data['3PA'].mean():.1f}")
        print(f"   Avg AST: {cluster_data['AST'].mean():.1f}")
        print(f"   Avg Pace: {cluster_data['Pace'].mean():.1f}")
        
        # Show era distribution
        era_counts = cluster_data['Era'].value_counts()
        print(f"   Era breakdown: {dict(era_counts)}")

def get_team_archetype(team_stats):
    """
    Predict which archetype a team belongs to
    """
    # Load the trained clustering model
    df = pd.read_csv("data/teams_with_archetypes.csv")
    
    # Get the most common archetype for this team (if it exists)
    if 'Team' in df.columns and 'Team' in team_stats:
        team_data = df[df['Team'] == team_stats['Team']]
        if len(team_data) > 0:
            return team_data['Archetype'].iloc[0]
    
    return "Unknown"

if __name__ == "__main__":
    print("🏀 NBA Offensive Archetypes Analysis")
    print("=" * 50)
    
    # Create archetypes
    df, model, scaler = create_offensive_archetypes()
    
    # Show some examples
    print(f"\n🔍 Sample Teams by Archetype:")
    for archetype in df['Archetype'].unique():
        sample_team = df[df['Archetype'] == archetype].iloc[0]
        print(f"   {archetype}: {sample_team['Team']} ({sample_team['Year']})")
    
    print(f"\n✅ Clustering analysis complete!")
    print(f"🎯 Found {df['Archetype'].nunique()} distinct offensive styles")
