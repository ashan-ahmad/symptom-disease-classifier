# 🩺 REMDX — Data Pipeline & Model Training

> **Recommendation Engine for Medical Diagnosis (REMDX)**
> A symptom-based disease prediction system powered by an Artificial Neural Network (ANN).

---

## 📋 Overview

REMDX is a Final Year Project (FYP) that predicts diseases from a set of **53 binary symptom inputs** across **10 disease classes**. This repository contains the complete data pipeline — from raw medical source texts to a trained Keras model — along with per-source test results and evaluation artifacts.

### Disease Classes

| # | Disease |
|---|---------|
| 1 | Malaria |
| 2 | Dengue |
| 3 | Typhoid |
| 4 | Tuberculosis |
| 5 | Common Cold |
| 6 | Pneumonia |
| 7 | Diabetes |
| 8 | Hypertension |
| 9 | Migraine |
| 10 | Cervical Spondylosis |

---

## 📁 Repository Structure

```
Data/
├── Raw/                          # Raw symptom text scraped from medical sources
│   ├── WHO.txt
│   ├── CDC.txt
│   ├── Mayo Clinic.txt
│   ├── Medlineplus.txt
│   └── Cleave land Clinic.txt
│
├── Scripts/                      # Python pipeline scripts
│   ├── csv_making.py             # Generate symptom CSV template (53 symptoms + prognosis)
│   ├── generate_test_data.py     # Generate synthetic binary test datasets
│   ├── combine_csvs.py           # Merge & interleave per-disease CSVs
│   ├── randomization.py          # Shuffle dataset rows for training fairness
│   ├── preprocessing.py          # StandardScaler + train/val split → .npy files
│   ├── model.py                  # ANN architecture definition (FYP_ANN_3)
│   ├── train.py                  # Model training loop
│   ├── test.py                   # Model evaluation with reports & confusion matrix
│   ├── dataset/
│   │   └── FinalCN_Dataset.csv   # Final combined & normalized dataset (~25 MB)
│   └── models/
│       └── final_model.keras     # Trained Keras model weights
│
├── Tests Results/                # Per-source evaluation outputs
│   ├── 1. WHO/
│   │   ├── WHO_Dataset.csv
│   │   ├── classification_report.csv
│   │   └── confusion_matrix.png
│   ├── 2. Mayo Clinic/
│   ├── 3. CDC/
│   ├── 4. Medlineplus/
│   └── 5. Cleave land Clinic/
│
└── README.md                     # ← You are here
```

---

## 🔬 Model Architecture — `FYP_ANN_3`

A fully-connected feedforward neural network built with the **Keras Functional API**:

```
Input (53 binary features)
    │
    ▼
Dense(16, ReLU) → Dropout(0.2)
    │
    ▼
Dense(16, ReLU) → Dropout(0.1)
    │
    ▼
Dense(16, ReLU) → Dropout(0.1)
    │
    ▼
Dense(10, Softmax) → Output (10 disease classes)
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 10 |
| Batch Size | 64 |
| Train / Val Split | 50 / 50 (stratified) |

---

## 🚀 Pipeline Workflow

The end-to-end pipeline follows this order:

```
1. Raw Text  ──►  2. CSV Template  ──►  3. Synthetic Data  ──►  4. Combine & Interleave
      │                  │                      │                         │
   Raw/*.txt       csv_making.py      generate_test_data.py        combine_csvs.py
                                                                         │
                                                                         ▼
                    7. Evaluate    ◄──  6. Train Model  ◄──  5. Shuffle & Preprocess
                         │                    │                      │
                      test.py             train.py          randomization.py
                                                            preprocessing.py
```

### Step-by-Step

1. **Raw Data Collection** — Symptom–disease mappings are manually extracted from authoritative medical sources (WHO, CDC, Mayo Clinic, Medlineplus, Cleveland Clinic) and saved as `.txt` files in `Raw/`.

2. **CSV Template Generation** — `csv_making.py` creates a CSV with 53 symptom columns + a `prognosis` column.

3. **Synthetic Dataset Generation** — `generate_test_data.py` produces binary datasets for each disease using configurable critical / partial symptom probabilities.

4. **Combine & Interleave** — `combine_csvs.py` merges per-disease CSVs with round-robin row interleaving for uniform class distribution.

5. **Shuffle & Preprocess** — `randomization.py` shuffles the combined dataset. `preprocessing.py` then applies `StandardScaler`, encodes labels with `LabelEncoder`, performs a 50/50 stratified train/validation split, and saves `.npy` artifacts + `scaler.pkl`.

6. **Train** — `train.py` loads preprocessed `.npy` files, builds the ANN via `model.py`, trains for 10 epochs, and saves `final_model.keras`.

7. **Evaluate** — `test.py` loads a saved model and scaler, runs predictions on a test CSV, and outputs:
   - `classification_report.csv` (precision, recall, F1-score)
   - `confusion_matrix.png` (heatmap visualization)

---

## ⚙️ Usage

### Prerequisites

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
```

### 1 — Preprocess the Dataset

```bash
python Scripts/preprocessing.py
```

### 2 — Train the Model

```bash
python Scripts/train.py
```

### 3 — Evaluate on a Test Set

Open `Scripts/test.py` and set the following paths:

```python
csv_path   = r"Path\to\your\test_dataset.csv"
model_path = r"Path\to\final_model.keras"
scaler_path = r"Path\to\scaler.pkl"
```

Then run:

```bash
python Scripts/test.py
```

Results (`classification_report.csv` and `confusion_matrix.png`) are saved alongside the test CSV.

---

## 📊 Data Sources

| Source | File | Description |
|--------|------|-------------|
| [WHO](https://www.who.int) | `Raw/WHO.txt` | World Health Organization disease fact sheets |
| [CDC](https://www.cdc.gov) | `Raw/CDC.txt` | U.S. Centers for Disease Control and Prevention |
| [Mayo Clinic](https://www.mayoclinic.org) | `Raw/Mayo Clinic.txt` | Mayo Clinic disease symptom references |
| [MedlinePlus](https://medlineplus.gov) | `Raw/Medlineplus.txt` | U.S. National Library of Medicine |
| [Cleveland Clinic](https://my.clevelandclinic.org) | `Raw/Cleave land Clinic.txt` | Cleveland Clinic health library |

---

## 🧪 Test Results

Each source has been independently tested against the trained model. Results are organized under `Tests Results/`:

| Source | Folder |
|--------|--------|
| WHO | `Tests Results/1. WHO/` |
| Mayo Clinic | `Tests Results/2. Mayo Clinic/` |
| CDC | `Tests Results/3. CDC/` |
| Medlineplus | `Tests Results/4. Medlineplus/` |
| Cleveland Clinic | `Tests Results/5. Cleave land Clinic/` |

Each folder contains:
- **`<Source>_Dataset.csv`** — The test dataset
- **`classification_report.csv`** — Precision, Recall, F1-Score per class
- **`confusion_matrix.png`** — Visual heatmap of predictions vs. ground truth

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| ML Utilities | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Serialization | joblib |

---

## 📄 License

This project was developed as a **Final Year Project (FYP)** under the Higher Education Commission (HEC).

---

<p align="center">
  <b>REMDX</b> — Bridging AI and Healthcare 🏥
</p>
