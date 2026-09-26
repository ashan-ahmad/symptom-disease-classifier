import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def main():
    # Resolve directory paths relative to this script
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    # Locate dataset
    dataset_candidates = [
        script_dir / "Dataset" / "FinalCN_Dataset.csv",
        script_dir / "dataset" / "FinalCN_Dataset.csv",
        base_dir / "Model" / "ANN3" / "Dataset" / "FinalCN_Dataset.csv",
        base_dir / "Dataset" / "FinalCN_Dataset.csv",
        base_dir / "FinalCN_Dataset.csv"
    ]
    dataset_path = None
    for candidate in dataset_candidates:
        if candidate.exists():
            dataset_path = candidate
            break

    if dataset_path is None:
        raise FileNotFoundError(
            f"Dataset not found. Checked: {[str(p) for p in dataset_candidates]}"
        )

    # Output directory for preprocessed files
    output_dir = script_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    data = pd.read_csv(dataset_path)

    # Separate features and labels (first 53 columns features, last column disease label)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Encode class labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_mapping = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
    print("Class mapping (class name -> numeric code):")
    for cls_name, code in class_mapping.items():
        print(f"  {cls_name}: {code}")

    # 50-50 Train-Validation Split (No test split)
    print("Performing 50/50 Train-Validation stratified split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.50, stratify=y_encoded, shuffle=True, random_state=42
    )

    # Standardize data using StandardScaler
    print("Standardizing features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Save preprocessed .npy files (Train & Validation only)
    np.save(output_dir / "X_train_scaled.npy", X_train_scaled)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val_scaled.npy", X_val_scaled)
    np.save(output_dir / "y_val.npy", y_val)

    print(f"Saved X_train_scaled.npy (shape: {X_train_scaled.shape})")
    print(f"Saved y_train.npy (shape: {y_train.shape})")
    print(f"Saved X_val_scaled.npy (shape: {X_val_scaled.shape})")
    print(f"Saved y_val.npy (shape: {y_val.shape})")
    print(f"Preprocessing complete. All files saved to: {output_dir}")

if __name__ == "__main__":
    main()
