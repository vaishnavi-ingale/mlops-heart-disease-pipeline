import os
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Heart Disease Risk Classification API",
    description="A production-ready MLOps serving engine providing real-time inference.",
    version="1.0.0"
)

# 1. Establish base search directories securely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlruns_search_base = os.path.join(BASE_DIR, "mlruns")

model_path = None

try:
    if os.path.exists(mlruns_search_base):
        # Walk through all internal subdirectories to find where the 'MLmodel' registry file lives
        for root, dirs, files in os.walk(mlruns_search_base):
            if "MLmodel" in files:
                model_path = root
                break  # Stop crawling files the moment we find the compiled binary directory

    # Validate that the lookup scanner discovered the target metrics
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not locate an active directory containing an MLmodel file inside: {mlruns_search_base}")
        
    print(f"Loading live production model engine from discovered path: {model_path}")
    model_pipeline = mlflow.sklearn.load_model(model_path)
    print("Live production model engine loaded successfully!")

except Exception as e:
    print(f"CRITICAL BOOT FAILURE: {str(e)}")
    raise e

# 2. Establish strict data validation boundaries using modern Pydantic V2 syntax
class PatientDataInput(BaseModel):
    age: float = Field(..., json_schema_extra={"example": 57.0})
    sex: float = Field(..., json_schema_extra={"example": 1.0})
    cp: float = Field(..., json_schema_extra={"example": 3.0})
    trestbps: float = Field(..., json_schema_extra={"example": 145.0})
    chol: float = Field(..., json_schema_extra={"example": 233.0})
    fbs: float = Field(..., json_schema_extra={"example": 1.0})
    restecg: float = Field(..., json_schema_extra={"example": 0.0})
    thalach: float = Field(..., json_schema_extra={"example": 150.0})
    exang: float = Field(..., json_schema_extra={"example": 0.0})
    oldpeak: float = Field(..., json_schema_extra={"example": 2.3})
    slope: float = Field(..., json_schema_extra={"example": 0.0})
    ca: float = Field(..., json_schema_extra={"example": 0.0})
    thal: float = Field(..., json_schema_extra={"example": 1.0})

# 3. Code the health check gateway endpoint
@app.get("/")
def read_root():
    return {"status": "healthy", "service": "Heart Disease Model API Serving Instance"}

# 4. Code the core inference predict endpoint matching the assignment task
@app.post("/predict")
def predict_heart_disease_risk(payload: PatientDataInput):
    try:
        # Convert incoming JSON dictionary seamlessly into a Pandas DataFrame shape expected by Scikit-Learn
        input_dict = payload.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Execute the full pipeline (Imputation -> Scaling -> Classification Weights)
        prediction = model_pipeline.predict(input_df)[0]
        probabilities = model_pipeline.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction])
        
        # Construct the official return schema payload response metrics
        return {
            "heart_disease_presence_risk": int(prediction),
            "prediction_label": "High Risk Detected" if prediction == 1 else "Normal Profile",
            "confidence_score": round(confidence, 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Processing Failure: {str(e)}")
