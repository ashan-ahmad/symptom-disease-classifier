import random
from pathlib import Path
import pandas as pd

def generate_binary_dataset(csv_path, critical_symptoms, partial_symptoms, prognosis_name, num_entries=1000, output_dir=None):
    # Load the symptom names from the CSV (first row only)
    df = pd.read_csv(csv_path, nrows=1)
    symptoms = list(df.columns[:-1])  # Exclude the last column (prognosis)

    # Ensure provided symptom lists exist in dataset
    critical_symptoms = [sym for sym in critical_symptoms if sym in symptoms]
    partial_symptoms = [sym for sym in partial_symptoms if sym in symptoms]

    # Create an empty DataFrame for the dataset
    data = []

    for _ in range(num_entries):
        entry = {symptom: 0 for symptom in symptoms}  # Default all to 0

        # Assign values for critical symptoms (0 to 10% as 1)
        for sym in critical_symptoms:
            entry[sym] = 1 if random.uniform(0, 1) <= random.uniform(0.0, 0.10) else 0

        # Assign values for partial symptoms (0 to 25% as 1)
        partial_threshold = random.uniform(0.0, 0.25)
        for sym in partial_symptoms:
            entry[sym] = 1 if random.uniform(0, 1) <= partial_threshold else 0

        # Assign the disease name in the prognosis column
        entry['prognosis'] = prognosis_name

        # Append entry to dataset
        data.append(entry)

    # Convert list to DataFrame
    final_df = pd.DataFrame(data)

    # Determine output path
    if output_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_dir = base_dir / "Model Multiple Tests" / "Datasets"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{prognosis_name}.csv"
    final_df.to_csv(output_csv, index=False)
    print(f"Binary dataset saved as {output_csv} with {num_entries} entries.")


if __name__ == "__main__":
    # Resolve path to all_symptoms.csv
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "Model Multiple Tests" / "all_symptoms.csv"
    if not csv_path.exists():
        csv_path = Path("all_symptoms.csv")

    critical_symptoms = ["Enter your critical symptoms"]
    partial_symptoms = ["Enter your partial symptoms"]
    prognosis_name = "Enter your disease name"

    generate_binary_dataset(csv_path, critical_symptoms, partial_symptoms, prognosis_name)
