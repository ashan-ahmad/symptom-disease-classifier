import os
import sys
from pathlib import Path
import pandas as pd

def combine_and_interleave_csvs(input_folder, output_file):
    """
    Read all CSV files in input_folder and interleave rows round-robin
    so that classes are uniformly distributed in output_file.
    """
    input_folder = Path(input_folder)
    output_file = Path(output_file)

    if not input_folder.exists():
        print(f"Warning: Input folder not found: {input_folder}")
        return False

    # List all CSV files in the folder
    csv_files = sorted([input_folder / f for f in os.listdir(input_folder) if f.endswith('.csv')])

    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return False

    print(f"Reading {len(csv_files)} CSV files from: {input_folder}")
    dataframes = [pd.read_csv(f) for f in csv_files]

    # Find the maximum number of rows among all CSVs
    max_rows = max(df.shape[0] for df in dataframes)

    # Interleave rows from all CSVs
    interleaved_rows = []
    for i in range(max_rows):
        for df in dataframes:
            if i < len(df):
                interleaved_rows.append(df.iloc[i])

    # Convert interleaved rows back into DataFrame
    combined_df = pd.DataFrame(interleaved_rows)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Combined CSV with interleaved rows saved as: {output_file}")
    print(f"Total rows: {len(combined_df)}, Columns: {len(combined_df.columns)}")
    return True

def main():
    base_dir = Path(__file__).resolve().parent.parent

    # Allow custom CLI arguments: python combine_csvs.py <input_folder> <output_file>
    if len(sys.argv) >= 3:
        input_folder = Path(sys.argv[1])
        output_file = Path(sys.argv[2])
        combine_and_interleave_csvs(input_folder, output_file)
        return

    # Default datasets present in 'Model Multiple Tests/Datasets'
    datasets_root = base_dir / "Model Multiple Tests" / "Datasets"
    known_targets = [
        ("WHO", "Sep", "WHO_Dataset.csv"),
        ("CDC", "Sep", "CDC_Dataset.csv"),
        ("Cleave land Clinic", "Sep", "Cleave_land_Clinic_Dataset.csv"),
        ("Mayo Clinic", "Sep", "Mayo_Clinic_Dataset.csv"),
        ("Medlineplus", "Sep", "Medlineplus_Dataset.csv"),
        ("Generated_0_10_0_25", "", "Generated_0_10_0_25_Dataset.csv")
    ]

    processed = 0
    for folder_name, subfolder, out_name in known_targets:
        in_path = datasets_root / folder_name / subfolder if subfolder else datasets_root / folder_name
        out_path = datasets_root / folder_name / out_name
        if in_path.exists() and any(f.endswith('.csv') for f in os.listdir(in_path)):
            if combine_and_interleave_csvs(in_path, out_path):
                processed += 1

    if processed == 0:
        # Fallback template run
        default_in = datasets_root / "WHO" / "Sep"
        default_out = datasets_root / "WHO" / "WHO_Dataset.csv"
        print("No predefined dataset folders populated yet.")
        print(f"Example usage: python combine_csvs.py \"{default_in}\" \"{default_out}\"")

if __name__ == "__main__":
    main()
