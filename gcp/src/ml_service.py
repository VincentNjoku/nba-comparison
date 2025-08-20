"""
GCP ML Service for NBA Era Analysis
Replaces local ML models with Vertex AI
"""
import logging
import uuid
import pandas as pd
import numpy as np
from google.cloud import aiplatform
from google.cloud.aiplatform import Model, Endpoint
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPMLService:
    def __init__(self):
        """Initialize Vertex AI client"""
        aiplatform.init(
            project=config.GCP_PROJECT_ID,
            location=config.VERTEX_AI_LOCATION
        )
        self.project_id = config.GCP_PROJECT_ID
        self.location = config.VERTEX_AI_LOCATION
        
    def train_and_deploy_model(self, training_data, model_name=None):
        """Train and deploy model to Vertex AI"""
        try:
            if not model_name:
                model_name = f"{config.MODEL_DISPLAY_NAME}-{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Starting model training: {model_name}")
            
            # Prepare training data
            X = training_data[['PTS', 'FGA', '3PA', '3P%', 'AST', 'TOV', '3P_Rate', 'Pace', 'ORtg']]
            y = training_data['Era']
            
            # Train model locally first (for demo purposes)
            # In production, you'd use Vertex AI Training
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy: {accuracy:.3f}")
            
            # Save model locally first
            import joblib
            model_path = f"gcp/models/{model_name}.pkl"
            joblib.dump(model, model_path)
            
            # Upload to Cloud Storage
            self._upload_model_to_gcs(model_path, model_name)
            
            # Deploy to Vertex AI
            endpoint = self._deploy_model_to_vertex_ai(model_name)
            
            logger.info(f"Model {model_name} deployed successfully to endpoint: {endpoint.name}")
            return endpoint.name
            
        except Exception as e:
            logger.error(f"Error training/deploying model: {e}")
            raise
    
    def _upload_model_to_gcs(self, local_model_path, model_name):
        """Upload trained model to Cloud Storage"""
        try:
            from google.cloud import storage
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(config.GCS_BUCKET)
            
            blob_name = f"{config.GCS_MODELS_PATH}/{model_name}.pkl"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_model_path)
            
            logger.info(f"Model uploaded to GCS: gs://{config.GCS_BUCKET}/{blob_name}")
            
        except Exception as e:
            logger.error(f"Error uploading model to GCS: {e}")
            raise
    
    def _deploy_model_to_vertex_ai(self, model_name):
        """Deploy model to Vertex AI Endpoint"""
        try:
            # Create model resource
            model = Model.upload(
                display_name=model_name,
                artifact_uri=f"gs://{config.GCS_BUCKET}/{config.GCS_MODELS_PATH}/{model_name}.pkl",
                serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest"
            )
            
            # Create endpoint
            endpoint = model.deploy(
                machine_type="n1-standard-2",
                accelerator_type=None,
                accelerator_count=None,
                min_replica_count=1,
                max_replica_count=3,
                traffic_split={"0": 100}
            )
            
            logger.info(f"Model deployed to endpoint: {endpoint.name}")
            return endpoint
            
        except Exception as e:
            logger.error(f"Error deploying to Vertex AI: {e}")
            raise
    
    def predict_era(self, team_stats, endpoint_name=None):
        """Predict NBA era using deployed Vertex AI model"""
        try:
            if not endpoint_name:
                # Use default endpoint
                endpoints = Endpoint.list()
                if not endpoints:
                    raise ValueError("No endpoints found. Please deploy a model first.")
                endpoint_name = endpoints[0].name
            
            endpoint = Endpoint(endpoint_name)
            
            # Prepare input data
            input_data = {
                'PTS': [team_stats['PTS']],
                'FGA': [team_stats['FGA']],
                '3PA': [team_stats['3PA']],
                '3P%': [team_stats['3P%']],
                'AST': [team_stats['AST']],
                'TOV': [team_stats['TOV']],
                '3P_Rate': [team_stats['3P_Rate']],
                'Pace': [team_stats['Pace']],
                'ORtg': [team_stats['ORtg']]
            }
            
            # Make prediction
            prediction = endpoint.predict(instances=[list(input_data.values())])
            
            # Extract results
            predicted_era = prediction.predictions[0][0]
            confidence = prediction.deployed_model_id  # In production, you'd get actual confidence
            
            logger.info(f"Prediction: {predicted_era} with confidence: {confidence}")
            
            return {
                'era': predicted_era,
                'confidence': confidence,
                'model_endpoint': endpoint_name
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # Fallback to local model if Vertex AI fails
            return self._fallback_local_prediction(team_stats)
    
    def _fallback_local_prediction(self, team_stats):
        """Fallback to local model if Vertex AI is unavailable"""
        try:
            import joblib
            import os
            
            # Try to load local model
            local_model_path = "models/era_classifier.pkl"
            if os.path.exists(local_model_path):
                model = joblib.load(local_model_path)
                
                # Prepare features
                features = [
                    team_stats['PTS'], team_stats['FGA'], team_stats['3PA'],
                    team_stats['3P%'], team_stats['AST'], team_stats['TOV'],
                    team_stats['3P_Rate'], team_stats['Pace'], team_stats['ORtg']
                ]
                
                prediction = model.predict([features])[0]
                confidence = 0.85  # Default confidence for fallback
                
                logger.info(f"Fallback prediction: {prediction}")
                return {
                    'era': prediction,
                    'confidence': confidence,
                    'model_endpoint': 'local_fallback'
                }
            else:
                raise FileNotFoundError("No local model available")
                
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return {
                'era': 'Unknown',
                'confidence': 0.0,
                'model_endpoint': 'error'
            }
    
    def get_model_metrics(self, endpoint_name=None):
        """Get model performance metrics"""
        try:
            if not endpoint_name:
                endpoints = Endpoint.list()
                if not endpoints:
                    return {}
                endpoint_name = endpoints[0].name
            
            endpoint = Endpoint(endpoint_name)
            
            # Get basic endpoint info
            metrics = {
                'endpoint_name': endpoint_name,
                'deployed_models': len(endpoint.list_models()),
                'traffic_split': endpoint.traffic_split,
                'create_time': endpoint.create_time.isoformat() if endpoint.create_time else None
            }
            
            logger.info(f"Retrieved metrics for endpoint: {endpoint_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {}
    
    def update_model_traffic(self, endpoint_name, model_id, traffic_percentage):
        """Update traffic split for A/B testing"""
        try:
            endpoint = Endpoint(endpoint_name)
            
            # Update traffic split
            traffic_split = {model_id: traffic_percentage}
            endpoint.update_traffic_split(traffic_split)
            
            logger.info(f"Updated traffic split: {traffic_split}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating traffic split: {e}")
            return False

if __name__ == "__main__":
    # Test the ML service
    ml_service = GCPMLService()
    print("GCP ML Service initialized successfully!") 