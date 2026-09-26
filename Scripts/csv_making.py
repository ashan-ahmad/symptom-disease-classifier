import csv
from pathlib import Path

def create_symptoms_csv(output_path=None):
    # List of 53 unique symptoms matching the FYP dataset
    symptoms = [
        "itching",
        "skin_rash",
        "continuous_sneezing",
        "shivering",
        "chills",
        "joint_pain",
        "acidity",
        "vomiting",
        "fatigue",
        "weight_loss",
        "restlessness",
        "lethargy",
        "irregular_sugar_level",
        "cough",
        "high_fever",
        "breathlessness",
        "sweating",
        "indigestion",
        "headache",
        "nausea",
        "loss_of_appetite",
        "pain_behind_the_eyes",
        "back_pain",
        "constipation",
        "abdominal_pain",
        "diarrhoea",
        "mild_fever",
        "swelled_lymph_nodes",
        "malaise",
        "blurred_and_distorted_vision",
        "phlegm",
        "chest_pain",
        "fast_heart_rate",
        "obesity",
        "excessive_hunger",
        "muscle_weakness",
        "stiff_neck",
        "swelling_joints",
        "movement_stiffness",
        "weakness_of_one_body_side",
        "toxic_look_(typhos)",
        "depression",
        "irritability",
        "muscle_pain",
        "altered_sensorium",
        "red_spots_over_body",
        "belly_pain",
        "watering_from_eyes",
        "increased_appetite",
        "polyuria",
        "rusty_sputum",
        "visual_disturbances",
        "painful_walking"
    ]

    # Create header by appending the 'prognosis' column
    header = symptoms + ["prognosis"]

    # Create a single default data row
    row = [1] * len(symptoms) + [""]

    if output_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_path = base_dir / "Model Multiple Tests" / "all_symptoms.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerow(row)

    print(f"CSV file '{output_path}' has been generated with a single row ({len(symptoms)} symptoms + prognosis).")
    return output_path

if __name__ == "__main__":
    create_symptoms_csv()
