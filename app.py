"""
NBA Era Comparison Machine - Streamlit Dashboard
Interactive web app for analyzing NBA offensive evolution across eras.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from ml_predictor import predict_era

# Page config
st.set_page_config(
    page_title="NBA Era Comparison Machine",
    page_icon="🏀",
    layout="wide"
)

# Color scheme
COLORS = {
    '1990s': '#1f77b4', '2000s': '#ff7f0e', '2010s': '#2ca02c', '2020s': '#d62728',
    'Traditional': '#1f77b4', 'Balanced': '#ff7f0e', 'Pace & Space': '#2ca02c', 'Modern 3-Point': '#d62728'
}

@st.cache_data
def load_data():
    """Load data files"""
    try:
        df = pd.read_csv("data/cleaned_stats.csv")
        df_archetypes = pd.read_csv("data/teams_with_archetypes.csv")
        return df, df_archetypes
    except:
        st.error("❌ Data files not found. Please run data_loader.py and cluster_analysis.py first.")
        return None, None

def era_overview(df):
    """Era overview tab"""
    st.header("🏠 NBA Era Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Era Distribution")
        era_counts = df['Era'].value_counts()
        fig = px.pie(values=era_counts.values, names=era_counts.index, title="Team Seasons by Era", color_discrete_map=COLORS)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 This shows the number of team seasons in our dataset for each era, not the number of NBA teams.")
        
        st.subheader("📊 Key Statistics by Era")
        era_stats = df.groupby('Era')[['PTS', '3PA', '3P_Rate', 'Pace']].mean().round(2)
        st.dataframe(era_stats, use_container_width=True)
        st.caption("💡 Pace = possessions per 48 minutes (higher = faster gameplay)")
    
    with col2:
        st.subheader("🏀 3-Point Evolution")
        fig = px.line(df.groupby('Year')['3PA'].mean().reset_index(), x='Year', y='3PA', 
                     title="3-Point Attempts Over Time", color_discrete_sequence=['#1f77b4'])
        fig.update_layout(xaxis_title="Year", yaxis_title="3-Point Attempts per Game")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📈 Scoring Trends")
        fig = px.line(df.groupby('Year')['PTS'].mean().reset_index(), x='Year', y='PTS', 
                     title="Points per Game Over Time", color_discrete_sequence=['#d62728'])
        fig.update_layout(xaxis_title="Year", yaxis_title="Points per Game")
        st.plotly_chart(fig, use_container_width=True)

def era_comparison(df):
    """Era comparison tab"""
    st.header("📊 Compare NBA Eras")
    
    col1, col2 = st.columns(2)
    with col1:
        era1 = st.selectbox("Select first era:", df['Era'].unique(), index=0)
    with col2:
        era2 = st.selectbox("Select second era:", df['Era'].unique(), index=1)
    
    if era1 != era2:
        era1_data = df[df['Era'] == era1]
        era2_data = df[df['Era'] == era2]
        metrics = ['PTS', '3PA', '3P_Rate', 'AST', 'Pace', 'ORtg']
        
        # Add metric explanations
        metric_explanations = {
            'PTS': 'Points per Game',
            '3PA': '3-Point Attempts per Game', 
            '3P_Rate': '3-Point Rate (3PA/FGA)',
            'AST': 'Assists per Game',
            'Pace': 'Possessions per 48 min',
            'ORtg': 'Offensive Rating'
        }
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🏀 {era1} Style")
            era1_avg = era1_data[metrics].mean().round(2)
            for metric in metrics:
                metric_name = metric_explanations.get(metric, metric)
                if metric == '3P_Rate':
                    st.metric(metric_name, f"{era1_avg[metric]:.1%}")
                elif metric == 'ORtg':
                    st.metric(metric_name, f"{era1_avg[metric]:.3f}")
                else:
                    st.metric(metric_name, f"{era1_avg[metric]:.1f}")
        
        with col2:
            st.subheader(f"🏀 {era2} Style")
            era2_avg = era2_data[metrics].mean().round(2)
            for metric in metrics:
                metric_name = metric_explanations.get(metric, metric)
                if metric == '3P_Rate':
                    st.metric(metric_name, f"{era2_avg[metric]:.1%}")
                elif metric == 'ORtg':
                    st.metric(metric_name, f"{era2_avg[metric]:.1%}")
                else:
                    st.metric(metric_name, f"{era2_avg[metric]:.1f}")
        
        # Charts
        st.subheader("📊 Era Comparison Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Large-Scale Metrics")
            large_metrics = ['PTS', '3PA', 'AST', 'TOV', 'Pace']
            large_data = [{'Metric': m, era1: era1_avg[m], era2: era2_avg[m]} for m in large_metrics if m in metrics]
            if large_data:
                large_df = pd.DataFrame(large_data)
                fig = px.bar(large_df, x='Metric', y=[era1, era2], barmode='group',
                            color_discrete_map={era1: COLORS[era1], era2: COLORS[era2]})
                fig.update_layout(title=f"{era1} vs {era2} - Large-Scale Metrics")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Small-Scale Metrics")
            small_metrics = ['3P_Rate', 'ORtg']
            small_data = [{'Metric': m, era1: era1_avg[m], era2: era2_avg[m]} for m in small_metrics if m in metrics]
            if small_data:
                small_df = pd.DataFrame(small_data)
                fig = px.bar(small_df, x='Metric', y=[era1, era2], barmode='group',
                            color_discrete_map={era1: COLORS[era1], era2: COLORS[era2]})
                fig.update_layout(title=f"{era1} vs {era2} - Small-Scale Metrics")
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Comparison Data")
        comparison_data = []
        for metric in metrics:
            comparison_data.append({
                'Metric': metric,
                f'{era1}': f"{era1_avg[metric]:.3f}",
                f'{era2}': f"{era2_avg[metric]:.3f}",
                'Difference': f"{era2_avg[metric] - era1_avg[metric]:+.3f}"
            })
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

def era_prediction(df):
    """ML era prediction tab"""
    st.header("🤖 Predict the Era")
    st.markdown("**Input team offensive stats to predict which NBA era they belong to**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Enter Team Stats")
        pts = st.number_input("Points per Game:", min_value=80.0, max_value=130.0, value=105.0, step=0.1)
        fga = st.number_input("Field Goal Attempts:", min_value=70.0, max_value=110.0, value=87.0, step=0.1)
        three_pa = st.number_input("3-Point Attempts:", min_value=5.0, max_value=45.0, value=20.0, step=0.1)
        three_pct = st.number_input("3-Point %:", min_value=0.20, max_value=0.45, value=0.35, step=0.01)
        ast = st.number_input("Assists:", min_value=15.0, max_value=35.0, value=24.0, step=0.1)
        tov = st.number_input("Turnovers:", min_value=10.0, max_value=25.0, value=15.0, step=0.1)
        
        if st.button("🔮 Predict Era"):
            three_rate = three_pa / fga if fga > 0 else 0
            pace = fga + 0.44 * (pts * 0.2) + tov
            ortg = pts / pace if pace > 0 else 0
            
            team_stats = {
                'PTS': pts, 'FGA': fga, '3PA': three_pa, '3P%': three_pct,
                'AST': ast, 'TOV': tov, '3P_Rate': three_rate, 'Pace': pace, 'ORtg': ortg
            }
            
            prediction = predict_era(team_stats)
            st.success(f"**Prediction: {prediction}**")
    
    with col2:
        st.subheader("📋 Sample Teams by Era")
        for era in df['Era'].unique():
            sample_team = df[df['Era'] == era].iloc[0]
            with st.expander(f"🏀 {era} Example"):
                st.write(f"**Team:** {sample_team['Team']} ({sample_team['Year']})")
                st.write(f"**PTS:** {sample_team['PTS']:.1f}")
                st.write(f"**3PA:** {sample_team['3PA']:.1f}")
                st.write(f"**AST:** {sample_team['AST']:.1f}")
                st.write(f"**Pace:** {sample_team['Pace']:.1f}")

def offensive_archetypes(df_archetypes):
    """Offensive archetypes tab"""
    st.header("🧠 NBA Offensive Archetypes")
    st.markdown("**Teams grouped by playing style regardless of era**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Archetype Distribution")
        archetype_counts = df_archetypes['Archetype'].value_counts()
        fig = px.pie(values=archetype_counts.values, names=archetype_counts.index, 
                     title="Teams by Offensive Archetype", color_discrete_map=COLORS)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Archetype Characteristics")
        archetype_stats = df_archetypes.groupby('Archetype')[['PTS', '3PA', '3P_Rate', 'AST', 'Pace']].mean().round(2)
        st.dataframe(archetype_stats, use_container_width=True)
    
    st.subheader("🏀 Archetype Breakdown")
    for archetype in df_archetypes['Archetype'].unique():
        with st.expander(f"📋 {archetype}"):
            archetype_data = df_archetypes[df_archetypes['Archetype'] == archetype]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Total Teams:** {len(archetype_data)}")
                st.write(f"**Years:** {archetype_data['Year'].min()} - {archetype_data['Year'].max()}")
            with col2:
                st.write(f"**Avg PTS:** {archetype_data['PTS'].mean():.1f}")
                st.write(f"**Avg 3PA:** {archetype_data['3PA'].mean():.1f}")
                st.write(f"**Avg AST:** {archetype_data['AST'].mean():.1f}")
            with col3:
                st.write(f"**3P Rate:** {archetype_data['3P_Rate'].mean():.1%}")
                st.write(f"**Pace:** {archetype_data['Pace'].mean():.1f}")
                st.write(f"**ORtg:** {archetype_data['ORtg'].mean():.3f}")
            
            era_dist = archetype_data['Era'].value_counts()
            clean_era_dist = {era: int(count) for era, count in era_dist.items()}
            st.write(f"**Era Distribution:** {clean_era_dist}")

def about():
    """About tab"""
    st.header("📁 About This Project")
    
    st.markdown("""
    ## 🏀 NBA Era Comparison Machine
    
    A comprehensive analysis of how NBA offensive strategy has evolved across different eras (1990s, 2000s, 2010s, 2020s).
    
    ### 🔧 Features
    
    **1. Era Analysis**
    - Compare offensive metrics across NBA eras
    - Visualize evolution of 3-point shooting, pace, and scoring
    - Interactive charts and data tables
    
    **2. Machine Learning Prediction**
    - Random Forest classifier with 76.3% accuracy
    - Predict which era a team belongs to based on offensive stats
    - Features: PTS, 3PA, 3P%, AST, TOV, 3P_Rate, Pace, ORtg
    
    **3. Offensive Archetypes**
    - K-means clustering identifies 4 distinct playing styles
    - Traditional, Balanced, Pace & Space, Modern 3-Point
    - Teams grouped by style regardless of era
    
    ### 🛠️ Technical Stack
    
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-learn (Random Forest, K-means)
    - **Visualization**: Plotly, Streamlit
    - **Data Source**: Basketball Reference team statistics (1990-2025)
    
    ### 🎯 Key Insights
    
    - **3-point shooting** is the biggest differentiator between eras
    - **Pace and scoring** have increased significantly over time
    - **Modern teams** focus on high-volume 3-point shooting
    - **Traditional teams** relied on inside scoring and mid-range shots
    
    ### 📊 Data Features
    
    - **Scoring**: PTS, FGA, FG%, 3PA, 3P%, 2PA, 2P%
    - **Playmaking**: AST, TOV, AST%, TOV%
    - **Efficiency**: ORtg, TS%, eFG%
    - **Pace**: Possessions per game, derived metrics
    
    ### 🚀 Getting Started
    
    1. Run `data_loader.py` to process NBA data
    2. Run `ml_predictor.py` to train the era classifier
    3. Run `cluster_analysis.py` to create offensive archetypes
    4. Run `streamlit run app.py` to launch the dashboard
    
    ---
    
    **Built for data science portfolios and NBA analytics enthusiasts.**
    
    *Data: Basketball Reference | ML: Scikit-learn | Dashboard: Streamlit*
    """)

def main():
    st.title("🏀 NBA Era Comparison Machine")
    st.markdown("**Analyze how NBA offensive strategy has evolved from the 1990s to the 2020s**")
    
    df, df_archetypes = load_data()
    if df is None:
        return
    
    st.sidebar.header("🎯 Navigation")
    tab = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Era Overview", "📊 Compare Eras", "🤖 Predict the Era", "🧠 Offensive Archetypes", "📁 About"]
    )
    
    if tab == "🏠 Era Overview":
        era_overview(df)
    elif tab == "📊 Compare Eras":
        era_comparison(df)
    elif tab == "🤖 Predict the Era":
        era_prediction(df)
    elif tab == "🧠 Offensive Archetypes":
        offensive_archetypes(df_archetypes)
    elif tab == "📁 About":
        about()

if __name__ == "__main__":
    main()
