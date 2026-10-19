import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_heart_disease_data():
    """
    Fetches the Heart Disease UCI dataset automatically using the UCI ML API,
    processes it into a unified DataFrame, and saves it cleanly to a data directory.
    """
    print("Initiating automated dataset retrieval from UCI Repository...")
    
    # Heart Disease unique ID is 45
    heart_disease = fetch_ucirepo(id=45)
    
    # Extract features and targets as pandas DataFrames
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Combine features and target into a single cohesive dataframe
    df = pd.concat([X, y], axis=1)
    
    # Define a clean target binary column name ('target' where 0=normal, 1+=disease presence)
    # The original dataset uses 0 for normal and 1,2,3,4 for varying degrees of heart disease.
    # The problem statement requires a binary target (presence vs absence).
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['num']) # Remove the old multi-class column
    
    # Establish a safe local directory to store the dataset
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, "heart_disease.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully acquired! Saved to: {output_path}")
    print(f"Dataset Shape: {df.shape}")

if __name__ == "__main__":
    download_heart_disease_data()
