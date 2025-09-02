# Cloud ML Cost Anomaly Detection

This repository provides an end-to-end machine learning and cloud architecture solution to detect and alert on cloud cost anomalies. The project combines **MLOps best practices** with **cloud-native infrastructure** for scalability, reproducibility, and observability.

## Key Features
- Automated ingestion of cloud billing exports
- Feature store (Feast) with online and offline support
- Training pipeline with MLflow experiment tracking
- Model serving using KServe on Kubernetes
- Canary releases and CI/CD with Argo Workflows and Argo CD
- Dashboards and alerts with Datadog

## High-Level Architecture
```
Billing Export → Ingestion Jobs → Data Lake
↓
Feature Store ↔ Training Pipeline → MLflow Registry
↓
Model Serving (KServe)
↓
Alerts and Dashboards (Datadog)
```

## Tech Stack
- Kubernetes (GKE/EKS)
- Terraform for infrastructure as code
- Argo Workflows and Argo CD
- MLflow for experiment tracking and registry
- Feast for feature management
- KServe for model serving
- Datadog for observability
- Python (scikit-learn, LightGBM, Prophet)

## Repository Structure

- infra/ # Terraform and K8s manifests
- pipelines/ # Argo workflows for ingestion, training, inference
- services/ # API and worker services
- ml/ # Data, features, training, serving code
- ops/ # CI/CD, monitoring, dashboards
- tests/ # Unit and integration tests


## Getting Started
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run unit tests: `make test`.
4. Deploy infrastructure with Terraform.
5. Apply Kubernetes manifests for Argo, MLflow, Feast, and KServe.
6. Start training and inference pipelines.

## Roadmap
- [ ] Local MVP with baseline anomaly detection model
- [ ] Deployment of MLflow, Feast, and Argo on Kubernetes
- [ ] Model serving with KServe
- [ ] Observability with Datadog
- [ ] Production hardening and CI/CD

## License
MIT License



















