"""
generate_eda_plots.py

Generates all EDA visualizations for the project and saves them to images/.
Run: python src/generate_eda_plots.py
"""

import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_preprocessing import load_data, clean_data, engineer_features

sns.set_style("whitegrid")
IMG_DIR = "images"

df = load_data()
df = clean_data(df)
df = engineer_features(df)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/{name}.png", dpi=120)
    plt.close(fig)
    print(f"Saved {name}.png")


# 1. Placement_Status distribution
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x="Placement_Status", hue="Placement_Status", palette="Set2", legend=False, ax=ax)
ax.set_title("Placement Status Distribution")
save(fig, "01_placement_distribution")

# 2. CGPA distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df["CGPA"], kde=True, color="steelblue", ax=ax)
ax.set_title("CGPA Distribution")
save(fig, "02_cgpa_distribution")

# 3. Attendance distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df["Attendance_Percentage"], kde=True, color="darkorange", ax=ax)
ax.set_title("Attendance Percentage Distribution")
save(fig, "03_attendance_distribution")

# 4. Study hours distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df["Study_Hours_Per_Day"], kde=True, color="seagreen", ax=ax)
ax.set_title("Study Hours Per Day Distribution")
save(fig, "04_study_hours_distribution")

# 5. Placement vs CGPA
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Placement_Status", y="CGPA", hue="Placement_Status", palette="Set2", legend=False, ax=ax)
ax.set_title("CGPA vs Placement Status")
save(fig, "05_placement_vs_cgpa")

# 6. Placement vs attendance
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Placement_Status", y="Attendance_Percentage", hue="Placement_Status", palette="Set2", legend=False, ax=ax)
ax.set_title("Attendance vs Placement Status")
save(fig, "06_placement_vs_attendance")

# 7. Placement vs internships
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Internships", hue="Placement_Status", palette="Set2", ax=ax)
ax.set_title("Internships vs Placement Status")
save(fig, "07_placement_vs_internships")

# 8. Placement vs projects
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Projects", hue="Placement_Status", palette="Set2", ax=ax)
ax.set_title("Projects vs Placement Status")
save(fig, "08_placement_vs_projects")

# 9. Placement vs technical skill score
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Placement_Status", y="Technical_Skill_Score", hue="Placement_Status", palette="Set2", legend=False, ax=ax)
ax.set_title("Technical Skill Score vs Placement Status")
save(fig, "09_placement_vs_technical_skill")

# 10. Placement vs communication score
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Placement_Status", y="Communication_Score", hue="Placement_Status", palette="Set2", legend=False, ax=ax)
ax.set_title("Communication Score vs Placement Status")
save(fig, "10_placement_vs_communication")

# 11. Correlation heatmap
numeric_cols = [
    "Age", "CGPA", "Attendance_Percentage", "Study_Hours_Per_Day", "Backlogs",
    "Internships", "Projects", "Certifications", "Communication_Score",
    "Technical_Skill_Score", "Aptitude_Score", "Total_Experience",
    "Skill_Average", "Project_Experience"
]
df_corr = df.copy()
df_corr["Placement_Status_Num"] = (df_corr["Placement_Status"] == "Placed").astype(int)
corr = df_corr[numeric_cols + ["Placement_Status_Num"]].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap (Numeric Features + Placement)")
save(fig, "11_correlation_heatmap")

# 12. Extra: Backlogs vs placement (useful additional insight)
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Backlogs", hue="Placement_Status", palette="Set2", ax=ax)
ax.set_title("Backlogs vs Placement Status")
save(fig, "12_placement_vs_backlogs")

# Print correlation of each feature with the target, sorted - used to write
# genuine, evidence-based insights instead of guessing.
print("\nCorrelation with Placement_Status_Num (sorted):")
print(corr["Placement_Status_Num"].drop("Placement_Status_Num").sort_values(ascending=False))
