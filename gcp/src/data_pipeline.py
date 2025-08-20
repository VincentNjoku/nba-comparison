"""
GCP Data Pipeline for NBA Era Analysis
Replaces CSV files with BigQuery and Cloud Storage
"""
import pandas as pd
import logging
from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPDataPipeline:
    def __init__(self):
        """Initialize GCP clients"""
        self.bq_client = bigquery.Client()
        self.storage_client = storage.Client()
        self.project_id = config.GCP_PROJECT_ID
        self.dataset_id = config.BIGQUERY_DATASET
        
    def create_dataset_and_tables(self):
        """Create BigQuery dataset and tables if they don't exist"""
        try:
            # Create dataset
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = config.GCP_REGION
            dataset = self.bq_client.create_dataset(dataset, exists_ok=True)
            logger.info(f"Dataset {self.dataset_id} ready")
            
            # Create tables
            self._create_team_stats_table()
            self._create_era_predictions_table()
            self._create_user_queries_table()
            
        except Exception as e:
            logger.error(f"Error creating dataset/tables: {e}")
            raise
    
    def _create_team_stats_table(self):
        """Create team stats table schema"""
        schema = [
            bigquery.SchemaField("Year", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("Team", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("PTS", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("FGA", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("3PA", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("3P_PCT", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("AST", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("TOV", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("3P_Rate", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("Pace", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("ORtg", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("Era", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("Created_At", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        table_id = f"{self.project_id}.{self.dataset_id}.{config.BIGQUERY_TABLES['team_stats']}"
        table = bigquery.Table(table_id, schema=schema)
        table = self.bq_client.create_table(table, exists_ok=True)
        logger.info(f"Table {config.BIGQUERY_TABLES['team_stats']} ready")
    
    def _create_era_predictions_table(self):
        """Create era predictions table schema"""
        schema = [
            bigquery.SchemaField("Prediction_ID", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("User_Input", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("Predicted_Era", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("Confidence", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("Model_Version", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("Created_At", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        table_id = f"{self.project_id}.{self.dataset_id}.{config.BIGQUERY_TABLES['era_predictions']}"
        table = bigquery.Table(table_id, schema=schema)
        table = self.bq_client.create_table(table, exists_ok=True)
        logger.info(f"Table {config.BIGQUERY_TABLES['era_predictions']} ready")
    
    def _create_user_queries_table(self):
        """Create user queries table schema"""
        schema = [
            bigquery.SchemaField("Query_ID", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("User_IP", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("Query_Type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("Query_Parameters", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("Response_Time_MS", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("Created_At", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        table_id = f"{self.project_id}.{self.dataset_id}.{config.BIGQUERY_TABLES['user_queries']}"
        table = bigquery.Table(table_id, schema=schema)
        table = self.bq_client.create_table(table, exists_ok=True)
        logger.info(f"Table {config.BIGQUERY_TABLES['user_queries']} ready")
    
    def upload_csv_to_bigquery(self, csv_file_path, table_name):
        """Upload CSV data to BigQuery"""
        try:
            # Read CSV
            df = pd.read_csv(csv_file_path)
            
            # Add timestamp
            df['Created_At'] = pd.Timestamp.now()
            
            # Upload to BigQuery
            table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                autodetect=True
            )
            
            job = self.bq_client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            job.result()  # Wait for job to complete
            
            logger.info(f"Successfully uploaded {len(df)} rows to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to BigQuery: {e}")
            return False
    
    def query_team_stats(self, era=None, team=None, year=None):
        """Query team stats from BigQuery"""
        try:
            query = f"""
            SELECT * FROM `{self.project_id}.{self.dataset_id}.{config.BIGQUERY_TABLES['team_stats']}`
            WHERE 1=1
            """
            
            if era:
                query += f" AND Era = '{era}'"
            if team:
                query += f" AND Team = '{team}'"
            if year:
                query += f" AND Year = {year}"
            
            query += " ORDER BY Year DESC, Team"
            
            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"Retrieved {len(df)} records from BigQuery")
            return df
            
        except Exception as e:
            logger.error(f"Error querying BigQuery: {e}")
            return pd.DataFrame()
    
    def log_prediction(self, prediction_id, user_input, predicted_era, confidence, model_version):
        """Log prediction to BigQuery for analytics"""
        try:
            data = {
                'Prediction_ID': [prediction_id],
                'User_Input': [str(user_input)],
                'Predicted_Era': [predicted_era],
                'Confidence': [confidence],
                'Model_Version': [model_version],
                'Created_At': [pd.Timestamp.now()]
            }
            
            df = pd.DataFrame(data)
            table_id = f"{self.project_id}.{self.dataset_id}.{config.BIGQUERY_TABLES['era_predictions']}"
            
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                autodetect=True
            )
            
            job = self.bq_client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            job.result()
            
            logger.info(f"Logged prediction {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            return False

if __name__ == "__main__":
    # Test the pipeline
    pipeline = GCPDataPipeline()
    pipeline.create_dataset_and_tables()
    print("GCP Data Pipeline initialized successfully!") 