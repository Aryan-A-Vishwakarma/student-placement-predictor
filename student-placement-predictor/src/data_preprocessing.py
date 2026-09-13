"""
data_preprocessing.py

Handles:
1. Loading the raw dataset
2. Cleaning (duplicates, missing values)
3. Feature engineering
4. Building a preprocessing pipeline (scaling + encoding) for modeling

Kept as reusable functions so both the notebook and model_training.py
can import and use the exact same logic (no copy-pasted code, no
inconsistency between notebook and script).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path="data/student_placement_data.csv"):
    """Load the raw CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df):
    """
    Clean the raw dataset.

    Decisions and why:
    - Duplicate rows are dropped entirely. A duplicate means the exact same
      student record appears twice, which would let the model "see" that
      student's outcome more than once and bias training.
    - Missing numeric values (CGPA, Attendance_Percentage, Communication_Score)
      are filled with the column MEDIAN, not the mean. Median is more robust
      to any skew/outliers, and with only ~2-3% missing in each column,
      imputing rather than dropping rows preserves valuable data.
    - Missing categorical values (Extracurricular_Activities) are filled with
      the column MODE (most frequent value), the standard simple approach
      when only a small fraction of values are missing.
    """
    df = df.drop_duplicates().copy()

    numeric_cols_with_na = ["CGPA", "Attendance_Percentage", "Communication_Score"]
    for col in numeric_cols_with_na:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols_with_na = ["Extracurricular_Activities"]
    for col in categorical_cols_with_na:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df.reset_index(drop=True)


def engineer_features(df):
    """
    Add engineered features that combine existing information in a way
    that's easier for a model (and a human) to use directly.

    - Total_Experience: internships done during college + whether the
      student had a placement-relevant internship before this cycle.
      Both pieces of info are known BEFORE the placement decision, so
      this does not leak future information.
    - Skill_Average: average of communication and technical skill scores,
      a simple combined "soft + hard skill" signal.
    - Project_Experience: combines hands-on project count with internship
      count as one "practical experience" number.
    """
    df = df.copy()
    df["Total_Experience"] = df["Internships"] + (df["Previous_Internship"] == "Yes").astype(int)
    df["Skill_Average"] = (df["Technical_Skill_Score"] + df["Communication_Score"]) / 2
    df["Project_Experience"] = df["Projects"] + df["Internships"]
    return df


def get_feature_columns():
    """
    The exact set of columns used as model inputs.
    Student_ID is excluded (it's just an identifier, not a real feature -
    using it would let the model "memorize" individual students).
    Placement_Status is excluded because it's the target, not an input.
    """
    numeric_features = [
        "Age", "CGPA", "Attendance_Percentage", "Study_Hours_Per_Day",
        "Backlogs", "Internships", "Projects", "Certifications",
        "Communication_Score", "Technical_Skill_Score", "Aptitude_Score",
        "Total_Experience", "Skill_Average", "Project_Experience",
    ]
    categorical_features = [
        "Gender", "Extracurricular_Activities", "Previous_Internship",
    ]
    return numeric_features, categorical_features


def build_preprocessor():
    """
    Builds a ColumnTransformer that:
    - Scales numeric features with StandardScaler (needed for Logistic
      Regression, which is sensitive to feature scale; harmless for
      Random Forest which ignores scale).
    - One-hot encodes categorical features (converts text categories like
      'Male'/'Female' into 0/1 columns models can actually use).

    IMPORTANT: this preprocessor must be FIT ONLY on training data
    (done automatically when placed inside an sklearn Pipeline and
    called with .fit() on X_train). This avoids data leakage - the
    scaler/encoder must never "see" the test set before evaluation.
    """
    numeric_features, categorical_features = get_feature_columns()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ])
    return preprocessor


def prepare_dataset(path="data/student_placement_data.csv"):
    """
    Full pipeline: load -> clean -> engineer features -> split into X, y.
    Returns X (features DataFrame) and y (target Series, 1=Placed, 0=Not Placed).
    """
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)

    numeric_features, categorical_features = get_feature_columns()
    X = df[numeric_features + categorical_features]
    y = (df["Placement_Status"] == "Placed").astype(int)

    return X, y, df
