# End-to-End MLOps Pipeline: Heart Disease Risk Classification

A production-grade, highly scalable, and reproducible machine learning solution built using enterprise MLOps best practices. The architecture handles everything automatically: from data ingestion and exploratory processing to cloud automation pipelines, containerised hosting layers, and infrastructure scaling.
 
---

# Production System Framework Architecture
The system lifecycle operates across a highly isolated data path to guarantee 100% operational reproducibility across distinct environments:
1. **Data Ingestion Script (`src/download_data.py`):** Fetches the raw feature metrics natively from the UCI Machine Learning API.
2. **Exploratory Visual Analytics (`src/eda.py`):** Assesses class balances and dependencies, generating high-resolution profiles directly to storage.
3. **Reproducible Modeling Pipeline (`src/train.py`):** Uses Scikit-Learn `ColumnTransformer` frameworks to inject medians for missing metrics, normalize continuous variables, and handle categorical variables cleanly. 
4. **Experiment Tracking Logs (`mlruns/`):** Evaluates competing Logistic Regression and Random Forest architectures through a 5-fold cross-validation setup, tracking performance metrics inside an MLflow database.
5. **Continuous Integration CI Platform (`.github/workflows/ci.yml`):** Automatically tests project code blocks via a cloud-hosted Linux server on every upload push.
6. **Isolated Serving API Container (`src/app.py` & `Dockerfile`):** Exposes a prediction endpoint via FastAPI using strict data schemas managed by Pydantic V2.
7. **Scale Cluster Orchestration Engine (`deployment.yaml`):** Scales twin active application instances (replicas) via Kubernetes load balancers for reliable traffic management.

---

# Step-by-Step Installation & Local Workspace Setup

Ensure you are sitting inside a native Linux Subsystem (`WSL 2`) directory space before firing up the setup loop.

# 1. Initialize the Isolated Sandbox Environment
```bash
# Re-create and activate your Python environment
python3 -m venv venv
source venv/bin/activate

# Upgrade the installer engine and batch install software dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

# 2. Execute the Data Retrieval & Visual Analytics Pipelines
```bash
# Automated dataset acquisition from UCI API repositories
python src/download_data.py

# Generate professional histograms and correlation matrix charts
python src/eda.py
```

# 3. Run Model Training & Review Experiment Registries
```bash
# Run cross-validation cycles and serialize the production model binary
python src/train.py

# Launch the visual tracking dashboard dashboard locally
mlflow ui
# (Open your regular browser to: http://localhost:5000 to view your metrics logs)
```

# 4. Run Automated Code Quality Checks
```bash
# Execute local unit test logic matrices
pytest tests/
```

---

# Containerization & Kubernetes Deployment Steps

# 1. Build and Run the API Container Locally via Docker
```bash
# Compile your API scripts and model weights into an isolated container layer
docker build --no-cache -t heart-disease-api:v1 .

# Launch the container service mapped live to system network port 8000
docker run -d -p 8000:8000 --name ml_serving_container heart-disease-api:v1

# Check the running process status tracker
docker ps
```
Interact with the running container via your web browser:`http://localhost:8000/docs`

# 2. Orchestrate and Scale Instances via Kubernetes
```bash
# Load your image directly inside the local Docker Desktop cluster context
docker build -t heart-disease-api:v1 .

# Apply the declarative scaling manifest sheet to launch twin replicas
kubectl apply -f deployment.yaml

# Monitor cluster pods to confirm status reads "Running"
kubectl get pods
```

---

# Real-Time Inference Testing (Sample Query Payload)
Once your API service is active (either inside Docker or your cluster nodes), you can test it directly by copying and pasting this network request command to fetch live predictions:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 57.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0, "chol": 233.0,
  "fbs": 1.0, "restecg": 0.0, "thalach": 150.0, "exang": 0.0,
  "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0
}'
```

# Expected Output Payload:
```json
{
  "heart_disease_presence_risk": 0,
  "prediction_label": "Normal Profile",
  "confidence_score": 0.8689
}
```

---

# Deliverables Checklist
- Automated Data Ingestion & EDA Modules (`src/download_data.py`, `src/eda.py`)
- Preprocessing Transformers & MLflow Tracking Suite (`src/train.py`)
- Automated Pytest Verification Script Suite (`tests/test_pipeline.py`)
- Automated Cloud Configuration Automation Pipeline (`.github/workflows/ci.yml`)
- FastAPI REST serving layer wrapped in production configurations (`src/app.py`, `Dockerfile`)
- Declarative Cluster Manifest Sheet configurations (`deployment.yaml`)
- Comprehensive Document Framework (`REPORT.md`)
