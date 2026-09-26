import sys
from pathlib import Path
import pandas as pd

def shuffle_csv(source_csv, output_csv=None, random_state=None):
    """
    Load a CSV file, randomly shuffle its rows, and save to output_csv.
    If output_csv is None, it overwrites source_csv.
    """
    source_path = Path(source_csv)
    if not source_path.exists():
        print(f"Error: File not found at {source_path}")
        return False

    if output_csv is None:
        output_path = source_path
    else:
        output_path = Path(output_csv)

    print(f"Loading CSV from: {source_path}")
    df = pd.read_csv(source_path)

    # Shuffle the rows randomly
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_shuffled.to_csv(output_path, index=False)

    print(f"Shuffled dataset saved as: {output_path}")
    print(f"Total rows shuffled: {df_shuffled.shape[0]}")
    return True

def main():
    base_dir = Path(__file__).resolve().parent.parent

    # CLI usage: python randomization.py <source_csv> [output_csv]
    if len(sys.argv) >= 2:
        src = Path(sys.argv[1])
        dst = Path(sys.argv[2]) if len(sys.argv) >= 3 else src
        shuffle_csv(src, dst)
        return

    # Check common datasets in 'Model Multiple Tests/Datasets'
    datasets_root = base_dir / "Model Multiple Tests" / "Datasets"
    candidates = [
        datasets_root / "CDC" / "CDC_Dataset.csv",
        datasets_root / "WHO" / "WHO_Dataset.csv",
        datasets_root / "Mayo Clinic" / "Mayo_Clinic_Dataset.csv",
        datasets_root / "Cleave land Clinic" / "Cleave_land_Clinic_Dataset.csv",
        datasets_root / "Medlineplus" / "Medlineplus_Dataset.csv"
    ]

    shuffled_any = False
    for candidate in candidates:
        if candidate.exists():
            shuffle_csv(candidate)
            shuffled_any = True

    if not shuffled_any:
        print("Usage:")
        print("  python randomization.py <path_to_csv> [optional_output_path]")
        print("Example:")
        print(f"  python randomization.py \"{datasets_root / 'CDC' / 'CDC_Dataset.csv'}\"")

if __name__ == "__main__":
    main()
