import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from pathlib import Path

# ===== USER: SET YOUR PATHS HERE =====
csv_path = r"Enter your test CSV path here"
model_path = r"Enter your model path here"
scaler_path = r"Enter your scaler.pkl path here"
# ======================================

# Load model and scaler
print(f"Loading model from: {model_path}")
model = load_model(model_path)

print(f"Loading scaler from: {scaler_path}")
scaler = joblib.load(scaler_path)

# Load test CSV
print(f"Loading test data from: {csv_path}")
new_data = pd.read_csv(csv_path)

# Separate features and labels
X_new = new_data.iloc[:, :-1].values
y_new = new_data.iloc[:, -1].values

# Scale features
X_new_scaled = scaler.transform(X_new)

# Encode labels
label_encoder = LabelEncoder()
y_new_encoded = label_encoder.fit_transform(y_new)

class_mapping = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
print("Class mapping (class name -> numeric code):")
print(class_mapping)

# Predict
predictions = model.predict(X_new_scaled)
predicted_labels = np.argmax(predictions, axis=1)

# Accuracy
test_accuracy = accuracy_score(y_new_encoded, predicted_labels)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Save classification report in same folder as the CSV
output_dir = Path(csv_path).parent
report = classification_report(y_new_encoded, predicted_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_csv_path = output_dir / "classification_report.csv"
report_df.to_csv(report_csv_path, index=True)
print(f"Classification report saved as: {report_csv_path}")

# Save confusion matrix in same folder as the CSV
cm = confusion_matrix(y_new_encoded, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
cm_png_path = output_dir / "confusion_matrix.png"
plt.savefig(cm_png_path)
plt.close()
print(f"Confusion matrix plot saved as: {cm_png_path}")
