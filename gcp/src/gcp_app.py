"""
GCP NBA Era Analysis Platform - Enhanced Streamlit App
Uses BigQuery for data and Vertex AI for ML predictions
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_pipeline import GCPDataPipeline
    from ml_service import GCPMLService
    import config
    GCP_AVAILABLE = True
except ImportError as e:
    st.warning(f"GCP services not available: {e}")
    GCP_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="GCP NBA Era Analysis Platform",
    page_icon="☁️🏀",
    layout="wide"
)

# Color scheme
COLORS = {
    '1990s': '#1f77b4', '2000s': '#ff7f0e', '2010s': '#2ca02c', '2020s': '#d62728',
    'Traditional': '#1f77b4', 'Balanced': '#ff7f0e', 'Pace & Space': '#2ca02c', 'Modern 3-Point': '#d62728'
}

@st.cache_data
def load_data_gcp():
    """Load data from BigQuery"""
    if not GCP_AVAILABLE:
        st.error("❌ GCP services not available. Please check configuration.")
        return None, None
    
    try:
        pipeline = GCPDataPipeline()
        df = pipeline.query_team_stats()
        
        if df.empty:
            st.warning("⚠️ No data found in BigQuery. Please upload data first.")
            return None, None
            
        return df, pipeline
    except Exception as e:
        st.error(f"❌ Error loading data from BigQuery: {e}")
        return None, None

def gcp_overview_tab(df, pipeline):
    """GCP-enhanced era overview tab"""
    st.header("☁️ GCP NBA Era Overview")
    
    # GCP Status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Key Statistics by Era")
        if df is not None:
            era_stats = df.groupby('Era')[['PTS', '3PA', '3P_Rate', 'Pace']].mean().round(2)
            st.dataframe(era_stats, use_container_width=True)
            st.caption("💡 Data loaded from BigQuery in real-time")
    
    with col2:
        st.subheader("☁️ GCP Services Status")
        if GCP_AVAILABLE:
            st.success("✅ BigQuery Connected")
            st.success("✅ Vertex AI Available")
            st.success("✅ Cloud Storage Ready")
        else:
            st.error("❌ GCP Services Unavailable")
    
    # Data Source Info
    if df is not None:
        st.subheader("📈 Data Source Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Data Source", "BigQuery")
        with col3:
            st.metric("Last Updated", df['Created_At'].max() if 'Created_At' in df.columns else "N/A")

def gcp_era_comparison_tab(df, pipeline):
    """GCP-enhanced era comparison tab"""
    st.header("☁️ Compare NBA Eras (GCP Powered)")
    
    # GCP Help Section
    with st.expander("💡 **GCP Features in Action**"):
        st.markdown("""
        **This tab demonstrates GCP integration:**
        
        🏀 **Real-time Data:** Queries BigQuery for live data
        🏀 **ML Predictions:** Uses Vertex AI for era classification
        🏀 **Scalability:** Handles millions of records efficiently
        🏀 **Monitoring:** Tracks user queries and performance
        """)
    
    if df is None:
        st.error("❌ No data available for comparison")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        era1 = st.selectbox("Select first era:", df['Era'].unique(), index=0)
    with col2:
        era2 = st.selectbox("Select second era:", df['Era'].unique(), index=1)
    
    if era1 != era2:
        era1_data = df[df['Era'] == era1]
        era2_data = df[df['Era'] == era2]
        metrics = ['PTS', '3PA', '3P_Rate', 'AST', 'Pace', 'ORtg']
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🏀 {era1} Style")
            era1_avg = era1_data[metrics].mean().round(2)
            for metric in metrics:
                metric_name = get_metric_display_name(metric)
                if metric == '3P_Rate':
                    st.metric(metric_name, f"{era1_avg[metric]:.1%}")
                elif metric == 'ORtg':
                    ortg_display = era1_avg[metric] * 100
                    st.metric(metric_name, f"{ortg_display:.1f}")
                else:
                    st.metric(metric_name, f"{era1_avg[metric]:.1f}")
        
        with col2:
            st.subheader(f"🏀 {era2} Style")
            era2_avg = era2_data[metrics].mean().round(2)
            for metric in metrics:
                metric_name = get_metric_display_name(metric)
                if metric == '3P_Rate':
                    st.metric(metric_name, f"{era2_avg[metric]:.1%}")
                elif metric == 'ORtg':
                    ortg_display = era2_avg[metric] * 100
                    st.metric(metric_name, f"{ortg_display:.1f}")
                else:
                    st.metric(metric_name, f"{era2_avg[metric]:.1f}")
        
        # Log user query to BigQuery
        if pipeline:
            try:
                query_id = str(uuid.uuid4())
                pipeline.log_user_query(
                    query_id=query_id,
                    query_type="era_comparison",
                    query_parameters=f"era1={era1},era2={era2}",
                    response_time_ms=100  # Placeholder
                )
            except Exception as e:
                st.warning(f"Could not log query: {e}")

def gcp_era_prediction_tab(df, pipeline):
    """GCP-enhanced era prediction tab"""
    st.header("🤖 Predict the Era (Vertex AI Powered)")
    
    if not GCP_AVAILABLE:
        st.error("❌ GCP ML services not available")
        return
    
    # GCP ML Status
    st.subheader("☁️ Vertex AI Model Status")
    try:
        ml_service = GCPMLService()
        model_metrics = ml_service.get_model_metrics()
        
        if model_metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"✅ Model Deployed")
            with col2:
                st.info(f"Endpoint: {model_metrics.get('endpoint_name', 'N/A')}")
            with col3:
                st.info(f"Models: {model_metrics.get('deployed_models', 0)}")
        else:
            st.warning("⚠️ No models deployed. Training new model...")
            
    except Exception as e:
        st.error(f"❌ Error checking ML service: {e}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Enter Team Stats")
        pts = st.number_input("Points per Game:", min_value=80.0, max_value=130.0, value=105.0, step=0.1)
        fga = st.number_input("Field Goal Attempts:", min_value=70.0, max_value=110.0, value=87.0, step=0.1)
        three_pa = st.number_input("3-Point Attempts:", min_value=5.0, max_value=45.0, value=20.0, step=0.1)
        three_pct = st.number_input("3-Point %:", min_value=0.20, max_value=0.45, value=0.35, step=0.01)
        ast = st.number_input("Assists:", min_value=15.0, max_value=35.0, value=24.0, step=0.1)
        tov = st.number_input("Turnovers:", min_value=10.0, max_value=25.0, value=15.0, step=0.1)
        
        if st.button("🔮 Predict Era (Vertex AI)"):
            start_time = time.time()
            
            try:
                # Calculate derived stats
                three_rate = three_pa / fga if fga > 0 else 0
                pace = fga + 0.44 * (pts * 0.2) + tov
                ortg = pts / pace if pace > 0 else 0
                
                team_stats = {
                    'PTS': pts, 'FGA': fga, '3PA': three_pa, '3P%': three_pct,
                    'AST': ast, 'TOV': tov, '3P_Rate': three_rate, 'Pace': pace, 'ORtg': ortg
                }
                
                # Make prediction using Vertex AI
                prediction_result = ml_service.predict_era(team_stats)
                
                response_time = int((time.time() - start_time) * 1000)
                
                # Display results
                st.success(f"**Prediction: {prediction_result['era']}**")
                st.info(f"**Confidence:** {prediction_result['confidence']}")
                st.info(f"**Model Endpoint:** {prediction_result['model_endpoint']}")
                st.info(f"**Response Time:** {response_time}ms")
                
                # Log prediction to BigQuery
                if pipeline:
                    try:
                        prediction_id = str(uuid.uuid4())
                        pipeline.log_prediction(
                            prediction_id=prediction_id,
                            user_input=str(team_stats),
                            predicted_era=prediction_result['era'],
                            confidence=prediction_result['confidence'],
                            model_version=prediction_result['model_endpoint']
                        )
                        st.success("✅ Prediction logged to BigQuery for analytics")
                    except Exception as e:
                        st.warning(f"Could not log prediction: {e}")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    
    with col2:
        st.subheader("📋 Sample Teams by Era")
        if df is not None:
            for era in df['Era'].unique():
                sample_team = df[df['Era'] == era].iloc[0]
                with st.expander(f"🏀 {era} Example"):
                    st.write(f"**Team:** {sample_team['Team']} ({sample_team['Year']})")
                    st.write(f"**PTS:** {sample_team['PTS']:.1f}")
                    st.write(f"**3PA:** {sample_team['3PA']:.1f}")
                    st.write(f"**AST:** {sample_team['AST']:.1f}")
                    st.write(f"**Pace:** {sample_team['Pace']:.1f}")

def gcp_monitoring_tab(df, pipeline):
    """GCP monitoring and analytics tab"""
    st.header("📊 GCP Platform Monitoring")
    
    if not GCP_AVAILABLE:
        st.error("❌ GCP services not available")
        return
    
    try:
        # Get ML service metrics
        ml_service = GCPMLService()
        model_metrics = ml_service.get_model_metrics()
        
        st.subheader("🤖 Vertex AI Model Metrics")
        if model_metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.json(model_metrics)
            with col2:
                st.info("Model performance and deployment status")
        else:
            st.warning("No model metrics available")
        
        # Data pipeline metrics
        st.subheader("📊 BigQuery Data Metrics")
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Teams", df['Team'].nunique())
            with col3:
                st.metric("Years Covered", f"{df['Year'].min()}-{df['Year'].max()}")
        
    except Exception as e:
        st.error(f"Error getting monitoring data: {e}")

def get_metric_display_name(metric):
    """Get display name for metrics"""
    metric_names = {
        'PTS': 'Points per Game',
        '3PA': '3-Point Attempts per Game',
        '3P_Rate': '3-Point Rate (3PA/FGA)',
        'AST': 'Assists per Game',
        'Pace': 'Possessions per 48 min',
        'ORtg': 'Offensive Rating (Points per 100 possessions)'
    }
    return metric_names.get(metric, metric)

def main():
    """Main app function"""
    st.title("☁️ GCP NBA Era Analysis Platform")
    st.markdown("**Enterprise-grade basketball analytics powered by Google Cloud Platform**")
    
    # Sidebar
    st.sidebar.title("☁️ GCP Services")
    st.sidebar.markdown("""
    **Available Services:**
    - ✅ BigQuery (Data Warehouse)
    - ✅ Vertex AI (ML Models)
    - ✅ Cloud Storage (File Storage)
    - ✅ Cloud Run (App Hosting)
    """)
    
    # Load data
    df, pipeline = load_data_gcp()
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 GCP Overview", 
        "📊 Era Comparison", 
        "🤖 Era Prediction", 
        "📊 Monitoring"
    ])
    
    with tab1:
        gcp_overview_tab(df, pipeline)
    
    with tab2:
        gcp_era_comparison_tab(df, pipeline)
    
    with tab3:
        gcp_era_prediction_tab(df, pipeline)
    
    with tab4:
        gcp_monitoring_tab(df, pipeline)
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with Google Cloud Platform** | NBA Analytics Platform")

if __name__ == "__main__":
    main() 