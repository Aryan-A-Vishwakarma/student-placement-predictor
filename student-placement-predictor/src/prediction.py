"""
prediction.py

Loads the trained model pipeline and provides a function to predict
placement outcome + probability for a NEW student's details.
"""

import pickle
import pandas as pd

from data_preprocessing import get_feature_columns


def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["pipeline"], saved["model_name"]


def predict_new_student(student_info, model_path="models/model.pkl"):
    """
    Predicts placement outcome for one new student.

    Parameters
    ----------
    student_info : dict
        Must contain the following keys:
        Age, CGPA, Attendance_Percentage, Study_Hours_Per_Day, Backlogs,
        Internships, Projects, Certifications, Communication_Score,
        Technical_Skill_Score, Aptitude_Score, Gender,
        Extracurricular_Activities, Previous_Internship

    Returns
    -------
    dict with keys: "prediction" ("Placed"/"Not Placed") and
    "probability" (float, 0-1, model's estimated probability of placement).
    """
    pipeline, _ = load_model(model_path)

    df = pd.DataFrame([student_info])

    # Recreate the same engineered features used during training.
    # (Must match data_preprocessing.engineer_features exactly, or the
    # model will receive inputs shaped differently than it was trained on.)
    df["Total_Experience"] = df["Internships"] + (df["Previous_Internship"] == "Yes").astype(int)
    df["Skill_Average"] = (df["Technical_Skill_Score"] + df["Communication_Score"]) / 2
    df["Project_Experience"] = df["Projects"] + df["Internships"]

    numeric_features, categorical_features = get_feature_columns()
    X_new = df[numeric_features + categorical_features]

    prediction = pipeline.predict(X_new)[0]
    probability = pipeline.predict_proba(X_new)[0, 1]  # P(Placed)

    return {
        "prediction": "Placed" if prediction == 1 else "Not Placed",
        "probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    # Example usage
    example_student = {
        "Age": 21,
        "CGPA": 8.2,
        "Attendance_Percentage": 88,
        "Study_Hours_Per_Day": 4.5,
        "Backlogs": 0,
        "Internships": 2,
        "Projects": 3,
        "Certifications": 2,
        "Communication_Score": 78,
        "Technical_Skill_Score": 82,
        "Aptitude_Score": 75,
        "Gender": "Female",
        "Extracurricular_Activities": "Yes",
        "Previous_Internship": "Yes",
    }

    result = predict_new_student(example_student)
    print(f"Prediction: {result['prediction']}")
    print(f"Estimated probability of placement: {result['probability'] * 100:.1f}%")
    print("\nNote: this is a model estimate based on patterns in synthetic training")
    print("data, NOT a guarantee of a real student's actual placement outcome.")
