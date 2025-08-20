# GCP NBA Era Analysis Platform - Infrastructure as Code
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Configure the Google Provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "bigquery.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "dataflow.googleapis.com"
  ])
  
  service = each.value
  disable_dependent_services = false
}

# BigQuery Dataset
resource "google_bigquery_dataset" "nba_analytics" {
  dataset_id  = "nba_analytics"
  description = "NBA team statistics and analytics data"
  location    = var.region
  
  labels = {
    environment = var.environment
    project     = "nba-analytics"
  }
}

# BigQuery Tables
resource "google_bigquery_table" "team_stats" {
  dataset_id = google_bigquery_dataset.nba_analytics.dataset_id
  table_id   = "team_stats"
  
  schema = file("${path.module}/schemas/team_stats.json")
  
  labels = {
    environment = var.environment
    table_type  = "fact"
  }
}

resource "google_bigquery_table" "era_predictions" {
  dataset_id = google_bigquery_dataset.nba_analytics.dataset_id
  table_id   = "era_predictions"
  
  schema = file("${path.module}/schemas/era_predictions.json")
  
  labels = {
    environment = var.environment
    table_type  = "fact"
  }
}

resource "google_bigquery_table" "user_queries" {
  dataset_id = google_bigquery_dataset.nba_analytics.dataset_id
  table_id   = "user_queries"
  
  schema = file("${path.module}/schemas/user_queries.json")
  
  labels = {
    environment = var.environment
    table_type  = "fact"
  }
}

# Cloud Storage Bucket
resource "google_storage_bucket" "nba_data" {
  name          = "${var.project_id}-nba-data"
  location      = var.region
  force_destroy = var.environment == "dev"
  
  uniform_bucket_level_access = true
  
  labels = {
    environment = var.environment
    purpose     = "nba-analytics"
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud Storage Bucket for ML Models
resource "google_storage_bucket" "ml_models" {
  name          = "${var.project_id}-ml-models"
  location      = var.region
  force_destroy = var.environment == "dev"
  
  uniform_bucket_level_access = true
  
  labels = {
    environment = var.environment
    purpose     = "ml-models"
  }
}

# Vertex AI Workbench (for model development)
resource "google_notebooks_instance" "ml_development" {
  count        = var.environment == "dev" ? 1 : 0
  name         = "nba-ml-development"
  location     = var.zone
  machine_type = "n1-standard-4"
  
  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf-latest-cpu"
  }
  
  labels = {
    environment = var.environment
    purpose     = "ml-development"
  }
}

# Cloud Run Service
resource "google_cloud_run_service" "nba_analysis" {
  name     = "nba-era-analysis"
  location = var.region
  
  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/nba-analysis:latest"
        
        ports {
          container_port = 8080
        }
        
        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }
        
        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }
        
        env {
          name  = "GCP_REGION"
          value = var.region
        }
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
  
  labels = {
    environment = var.environment
    service     = "nba-analysis"
  }
}

# Cloud Build Trigger
resource "google_cloudbuild_trigger" "nba_analysis_build" {
  name        = "nba-analysis-build"
  description = "Build and deploy NBA analysis app"
  
  github {
    owner  = var.github_owner
    name   = var.github_repo
    push {
      branch = "main"
    }
  }
  
  filename = "cloudbuild.yaml"
  
  substitutions = {
    _REGION = var.region
  }
}

# IAM for Cloud Run
resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_service.nba_analysis.location
  service  = google_cloud_run_service.nba_analysis.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Service Account for the application
resource "google_service_account" "nba_analysis_sa" {
  account_id   = "nba-analysis-sa"
  display_name = "NBA Analysis Service Account"
  description  = "Service account for NBA analysis application"
}

# IAM bindings for the service account
resource "google_project_iam_member" "bigquery_user" {
  project = var.project_id
  role    = "roles/bigquery.user"
  member  = "serviceAccount:${google_service_account.nba_analysis_sa.email}"
}

resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.nba_analysis_sa.email}"
}

resource "google_project_iam_member" "aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.nba_analysis_sa.email}"
}

# Cloud Monitoring Dashboard
resource "google_monitoring_dashboard" "nba_analytics" {
  dashboard_json = file("${path.module}/monitoring/dashboard.json")
}

# Outputs
output "bigquery_dataset" {
  value = google_bigquery_dataset.nba_analytics.dataset_id
}

output "cloud_storage_bucket" {
  value = google_storage_bucket.nba_data.name
}

output "cloud_run_service_url" {
  value = google_cloud_run_service.nba_analysis.status[0].url
}

output "service_account_email" {
  value = google_service_account.nba_analysis_sa.email
} 