import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_exploratory_analysis():
    """
    Loads the downloaded dataset, conducts statistical verification,
    and exports professional evaluation plots for documentation.
    """
    print("Initiating Exploratory Data Analysis (EDA)...")
    
    # Define project path directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "heart_disease.csv")
    output_dir = os.path.join(base_dir, "screenshots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}. Run download_data.py first!")
        
    df = pd.read_csv(data_path)
    
    # Display basic diagnostic metrics to the terminal console
    print("\n--- Dataset Profile ---")
    print(df.info())
    print("\n--- Missing Value Count ---")
    print(df.isnull().sum())
    
    # Use standard professional styling for plots
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # PLOT 1: Class Balance Check
    # -------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, hue='target', palette='Set2', legend=False)
    plt.title("Class Distribution (0 = Normal, 1 = Heart Disease Presence)")
    plt.xlabel("Diagnosis Label")
    plt.ylabel("Patient Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_class_balance.png"), dpi=300)
    plt.close()
    print("Class balance plot exported successfully.")

    # -------------------------------------------------------------
    # PLOT 2: Feature Distribution (Age vs Blood Pressure)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['age'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Age Distribution Profile")
    axes[0].set_xlabel("Age (Years)")
    
    sns.histplot(df['trestbps'].dropna(), kde=True, ax=axes[1], color='salmon')
    axes[1].set_title("Resting Blood Pressure Profile")
    axes[1].set_xlabel("Blood Pressure (mm Hg)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_feature_distributions.png"), dpi=300)
    plt.close()
    print("Feature distribution plots exported successfully.")

    # -------------------------------------------------------------
    # PLOT 3: Linear Correlation Matrix Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 10))
    # Select only numeric data types to compute correlation safely
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Grid of Health Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_matrix.png"), dpi=300)
    plt.close()
    print("Linear correlation matrix exported successfully.")
    
    print(f"\nAll visual analytics saved safely inside: {output_dir}")

if __name__ == "__main__":
    run_exploratory_analysis()
