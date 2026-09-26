import os
import sys
from pathlib import Path
import numpy as np

# Ensure current script folder is in sys.path for importing model
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from model import build_model

def main():
    base_dir = script_dir.parent

    # Directory where preprocessed .npy files are stored
    prep_dir = base_dir / "Model" / "ANN3" / "Preprocessing"

    print(f"Loading preprocessed data from: {prep_dir}")
    X_train_scaled = np.load(prep_dir / "X_train_scaled.npy")
    y_train = np.load(prep_dir / "y_train.npy")
    X_val_scaled = np.load(prep_dir / "X_val_scaled.npy")
    y_val = np.load(prep_dir / "y_val.npy")

    print(f"X_train shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val_scaled.shape}, y_val shape:   {y_val.shape}")

    # Build model
    num_features = X_train_scaled.shape[1]
    num_classes = len(np.unique(np.concatenate([y_train, y_val])))
    print(f"Building model with {num_features} input features and {num_classes} output classes...")
    model = build_model(input_shape=(num_features,), num_classes=num_classes)

    # Train model with current parameters
    print("Training model (epochs=10, batch_size=64)...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val_scaled, y_val),
        verbose=1
    )

    # Save model in Model folder
    model_dir = base_dir / "Model"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "final_model.keras"

    # Also save in ANN3/Training folder if desired
    ann3_training_dir = base_dir / "Model" / "ANN3" / "Training"
    ann3_training_dir.mkdir(parents=True, exist_ok=True)
    ann3_save_path = ann3_training_dir / "final_model.keras"

    model.save(save_path)
    model.save(ann3_save_path)

    print(f"Training completed successfully.")
    print(f"Model saved to: {save_path}")
    print(f"Backup copy saved to: {ann3_save_path}")

if __name__ == "__main__":
    main()
