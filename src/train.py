import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def build_ml_pipeline():
    """
    Implements a production-grade ML training suite with Scikit-Learn and MLflow.
    Handles data processing, cross-validation, and automated experiment logging.
    """
    # 1. Define folder layout systems
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "heart_disease.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing core training data file at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 2. Separate independent features from target vector
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 3. Categorize feature columns for targeted processing
    # The UCI dataset blends continuous variables and encoded categorical numbers
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # 4. Construct robust preprocessing pipelines (Task 4 Reproducibility requirement)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Safely handles missing health logs using median
        ('scaler', StandardScaler())                    # Normalizes data features to stable variance metrics
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handles missing flags with categorical modes
        ('encoder', OneHotEncoder(handle_unknown='ignore'))   # Robust to unexpected live prediction entries
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    # 5. Split data into pristine training and evaluation folds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Initialize MLflow Experiment Tracking Name
    mlflow.set_experiment("Heart_Disease_Classifier_Experimentation")
    
    # 7. Model evaluation dictionaries
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    print("\n--- Starting Model Training Suite via MLflow Context Manager ---")
    
    for model_name, model_obj in models.items():
        # Open an isolated tracking record instance block for each run configuration
        with mlflow.start_run(run_name=model_name):
            print(f"\nEvaluating Architecture Model: {model_name}")
            
            # Combine preprocessor stage and algorithm step into a unified operational Pipeline
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model_obj)
            ])
            
            # Run Robust Cross Validation Metrics on training fold (Task 2)
            cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='accuracy')
            mean_cv_acc = np.mean(cv_scores)
            
            # Fit final pipeline cleanly
            full_pipeline.fit(X_train, y_train)
            
            # Compute core performance prediction outputs
            y_pred = full_pipeline.predict(X_test)
            y_proba = full_pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                "cv_mean_accuracy": mean_cv_acc,
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_roc_auc": roc_auc_score(y_test, y_proba)
            }
            
            # Print metrics summary straight to shell output logs
            for metric_k, metric_v in metrics.items():
                print(f" -> {metric_k}: {metric_v:.4f}")
            
            # --- MLflow Logging Operations (Task 3 & 4) ---
            # Log structural algorithmic parameters
            if hasattr(model_obj, 'n_estimators'):
                mlflow.log_param("n_estimators", model_obj.n_estimators)
                mlflow.log_param("max_depth", model_obj.max_depth)
            if hasattr(model_obj, 'max_iter'):
                mlflow.log_param("max_iter", model_obj.max_iter)
                
            # Log calculated evaluation metric matrices
            mlflow.log_metrics(metrics)
            
            # Log the complete pipeline artifact directly (Packaging the entire preprocessor state + model)
                        # Log the complete pipeline artifact directly with security exceptions handled
            mlflow.sklearn.log_model(
                full_pipeline, 
                artifact_path="model", 
                registered_model_name=model_name,
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]  # Authorises numpy metadata checks safely
            )
            print(f"Successfully tracked model configuration artifact inside MLflow registries.")


if __name__ == "__main__":
    build_ml_pipeline()
