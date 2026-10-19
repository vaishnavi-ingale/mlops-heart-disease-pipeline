import os
import pandas as pd
import pytest

def test_dataset_existence_and_structure():
    """
    Automated check ensuring the data ingestion system downloaded a 
    valid dataset containing the expected feature footprints.
    """
    # Locating the downloaded datafile asset path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "heart_disease.csv")
    
    # Assert 1: Confirm the download script ran and generated the file asset
    assert os.path.exists(data_path), f"Critical Error: heart_disease.csv dataset is missing at {data_path}"
    
    # Load dataset fold for structural assessment
    df = pd.read_csv(data_path)
    
    # Assert 2: Verify the data matrix holds rows and structural feature attributes
    assert len(df) > 0, "Dataset contains 0 active patient rows."
    
    # Assert 3: Confirm our ingestion pipeline successfully mapped the target into binary metrics
    assert 'target' in df.columns, "The label column mapping 'target' is missing."
    assert set(df['target'].unique()).issubset({0, 1}), "Target classification values are not strictly binary (0 or 1)."
    
    # Assert 4: Confirm critical base health parameter features are included
    expected_features = ['age', 'sex', 'chol', 'trestbps', 'thalach']
    for feature in expected_features:
        assert feature in df.columns, f"Essential model feature metric column '{feature}' is missing."
