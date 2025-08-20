"""
GCP Configuration for NBA Era Analysis Platform
"""
import os
from google.cloud import bigquery, storage, aiplatform

# GCP Project Configuration
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-project-id')
GCP_REGION = os.getenv('GCP_REGION', 'us-central1')
GCP_ZONE = os.getenv('GCP_ZONE', 'us-central1-a')

# BigQuery Configuration
BIGQUERY_DATASET = 'nba_analytics'
BIGQUERY_TABLES = {
    'team_stats': 'team_stats',
    'era_predictions': 'era_predictions',
    'user_queries': 'user_queries'
}

# Cloud Storage Configuration
GCS_BUCKET = f'{GCP_PROJECT_ID}-nba-data'
GCS_DATA_PATH = 'data'
GCS_MODELS_PATH = 'models'

# Vertex AI Configuration
VERTEX_AI_LOCATION = GCP_REGION
MODEL_DISPLAY_NAME = 'nba-era-classifier'
ENDPOINT_DISPLAY_NAME = 'nba-era-prediction-endpoint'

# Cloud Run Configuration
SERVICE_NAME = 'nba-era-analysis'
SERVICE_URL = f'https://{SERVICE_NAME}-{GCP_PROJECT_ID}.run.app'

# Data Processing Configuration
BATCH_SIZE = 1000
MAX_WORKERS = 4

# Monitoring Configuration
METRIC_PREFIX = 'custom.googleapis.com/nba_analysis'
LOG_LEVEL = 'INFO'

# Feature flags
ENABLE_REAL_TIME_PREDICTIONS = True
ENABLE_BATCH_PROCESSING = True
ENABLE_MONITORING = True 