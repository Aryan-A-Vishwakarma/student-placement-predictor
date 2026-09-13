"""
generate_dataset.py

Creates a SYNTHETIC student placement dataset for the
Student Performance & Placement Predictor project.

This data is artificially generated to LOOK realistic (it has believable
ranges and relationships between features and placement), but it is NOT
collected from real students. It exists purely to demonstrate a complete
data analysis + machine learning workflow.

Run this file once to (re)create data/student_placement_data.csv
    python src/generate_dataset.py
"""

import numpy as np
import pandas as pd

# Fixed seed = reproducibility. Anyone who runs this script gets the
# exact same "random" data every time.
np.random.seed(42)

N = 1200  # number of student records

# ---------------------------------------------------------------
# 1. Generate base features
# ---------------------------------------------------------------

student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, N + 1)]

gender = np.random.choice(["Male", "Female"], size=N, p=[0.58, 0.42])

age = np.random.randint(20, 23, size=N)

# CGPA: most students cluster around 6.5-8.5, few extremes
cgpa = np.clip(np.random.normal(loc=7.2, scale=0.9, size=N), 4.5, 10.0).round(2)

attendance = np.clip(np.random.normal(loc=80, scale=10, size=N), 40, 100).round(1)

study_hours = np.clip(np.random.normal(loc=3, scale=1.5, size=N), 0, 10).round(1)

# Backlogs: most students have 0, a smaller number have 1-3
backlogs = np.random.choice(
    [0, 1, 2, 3], size=N, p=[0.65, 0.20, 0.10, 0.05]
)

internships = np.random.choice([0, 1, 2, 3], size=N, p=[0.35, 0.35, 0.20, 0.10])

projects = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.10, 0.25, 0.30, 0.20, 0.15])

certifications = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.30, 0.30, 0.20, 0.12, 0.08])

communication_score = np.clip(np.random.normal(loc=65, scale=15, size=N), 20, 100).round(1)

technical_skill_score = np.clip(np.random.normal(loc=65, scale=15, size=N), 20, 100).round(1)

aptitude_score = np.clip(np.random.normal(loc=60, scale=18, size=N), 10, 100).round(1)

extracurricular = np.random.choice(["Yes", "No"], size=N, p=[0.45, 0.55])

previous_internship = np.random.choice(["Yes", "No"], size=N, p=[0.4, 0.6])

# ---------------------------------------------------------------
# 2. Build a "placement score" that realistically depends on
#    several features (with noise), then convert to Placed/Not Placed.
#    This avoids the target being trivially determined by one column.
# ---------------------------------------------------------------

score = (
    0.9 * cgpa
    + 0.03 * attendance
    + 0.25 * study_hours
    - 0.8 * backlogs
    + 0.9 * internships
    + 0.5 * projects
    + 0.3 * certifications
    + 0.02 * communication_score
    + 0.03 * technical_skill_score
    + 0.02 * aptitude_score
    + np.where(extracurricular == "Yes", 0.4, 0)
    + np.where(previous_internship == "Yes", 0.6, 0)
)

# Add random noise so placement isn't perfectly predictable
# (real life has randomness: interview mood, luck, company needs, etc.)
score += np.random.normal(loc=0, scale=1.8, size=N)

# Convert score into a probability using a logistic-style curve,
# then sample Placed/Not Placed from that probability.
prob_placed = 1 / (1 + np.exp(-(score - score.mean()) / 2.2))
placement_status = np.where(
    np.random.rand(N) < prob_placed, "Placed", "Not Placed"
)

# ---------------------------------------------------------------
# 3. Assemble the DataFrame
# ---------------------------------------------------------------

df = pd.DataFrame({
    "Student_ID": student_ids,
    "Gender": gender,
    "Age": age,
    "CGPA": cgpa,
    "Attendance_Percentage": attendance,
    "Study_Hours_Per_Day": study_hours,
    "Backlogs": backlogs,
    "Internships": internships,
    "Projects": projects,
    "Certifications": certifications,
    "Communication_Score": communication_score,
    "Technical_Skill_Score": technical_skill_score,
    "Aptitude_Score": aptitude_score,
    "Extracurricular_Activities": extracurricular,
    "Previous_Internship": previous_internship,
    "Placement_Status": placement_status,
})

# ---------------------------------------------------------------
# 4. Deliberately inject some messiness so the cleaning stage
#    has something real to demonstrate (this is common in real data).
# ---------------------------------------------------------------

# 4a. Add ~3% missing values scattered across a few realistic columns
rng = np.random.default_rng(42)
for col, frac in [
    ("CGPA", 0.02),
    ("Attendance_Percentage", 0.03),
    ("Communication_Score", 0.02),
    ("Extracurricular_Activities", 0.02),
]:
    n_missing = int(frac * N)
    missing_idx = rng.choice(df.index, size=n_missing, replace=False)
    df.loc[missing_idx, col] = np.nan

# 4b. Duplicate ~15 rows (simulates accidental double data-entry)
dup_rows = df.sample(n=15, random_state=42)
df = pd.concat([df, dup_rows], ignore_index=True)

# 4c. Shuffle so duplicates/missing aren't neatly at the end
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------
output_path = "data/student_placement_data.csv"
df.to_csv(output_path, index=False)

print(f"Dataset created: {output_path}")
print(f"Shape: {df.shape}")
print(f"Placement_Status distribution:\n{df['Placement_Status'].value_counts()}")
