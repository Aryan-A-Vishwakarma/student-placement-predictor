# 🎓 Student Performance & Placement Predictor

A complete, end-to-end machine learning project that analyzes student academic
and extracurricular data to understand which factors are associated with
campus placement, and builds classification models that predict whether a
student is likely to be placed.

## Overview

This project walks through a full data science workflow: understanding raw
data, cleaning it, exploring it visually, engineering useful features,
training multiple machine learning models, evaluating them honestly, and
using the best one to predict outcomes for new students — with both a
notebook and a simple interactive web app.

It was built as a learning-focused portfolio project, with an emphasis on
understanding *why* each step is done, not just running code.

## Problem Statement

Placement season is stressful, and students often wonder which of their
efforts (academics, projects, internships, soft skills) actually correlate
with getting placed. This project explores that question quantitatively and
builds a predictive tool around it.

## Objectives

- Explore relationships between student attributes and placement outcomes.
- Build and fairly compare multiple classification models.
- Avoid data leakage and produce a reproducible, evaluable pipeline.
- Package the result as a clean, understandable, portfolio-ready project.

## Dataset

**This dataset is synthetically generated** (see `src/generate_dataset.py`) —
it is **not** real student data collected from any institution.

Public datasets in this space (checked on Kaggle) with a comparable feature
set are, almost without exception, uploader-generated simulations themselves,
often without clear documentation of that fact. Rather than use an
undocumented "real-looking" dataset, this project generates its own data with
transparent, realistic feature-to-outcome relationships plus deliberately
injected missing values and duplicate rows, so the full cleaning workflow
could be demonstrated honestly.

**Important:** because the data is synthetic, results below demonstrate the
technical workflow correctly, but should not be interpreted as real-world
findings about what determines student placement.

### Features

| Column | Description |
|---|---|
| `Student_ID` | Unique identifier (excluded from modeling) |
| `Gender` | Male / Female |
| `Age` | Student age (20–22) |
| `CGPA` | Cumulative GPA, 0–10 scale |
| `Attendance_Percentage` | Overall attendance, 0–100% |
| `Study_Hours_Per_Day` | Average self-study hours per day |
| `Backlogs` | Number of pending/failed subjects |
| `Internships` | Number of internships completed |
| `Projects` | Number of academic/personal projects completed |
| `Certifications` | Number of relevant certifications earned |
| `Communication_Score` | Soft-skill score, 0–100 |
| `Technical_Skill_Score` | Technical proficiency score, 0–100 |
| `Aptitude_Score` | Logical/quantitative aptitude score, 0–100 |
| `Extracurricular_Activities` | Yes / No |
| `Previous_Internship` | Whether student had a prior internship |
| `Placement_Status` | **Target** — Placed / Not Placed |

Engineered features added during preprocessing: `Total_Experience`,
`Skill_Average`, `Project_Experience` (see Feature Engineering section below).

## Technologies Used

- Python, Pandas, NumPy
- Matplotlib, Seaborn (visualization)
- Scikit-learn (modeling, pipelines, evaluation)
- Jupyter Notebook
- Streamlit (optional interactive prediction app)

## Project Structure

```
student-placement-predictor/
│
├── data/
│   └── student_placement_data.csv       # Cleaned/raw synthetic dataset
│
├── notebooks/
│   └── student_placement_analysis.ipynb # Full analysis notebook (the "story")
│
├── src/
│   ├── generate_dataset.py              # Creates the synthetic dataset
│   ├── data_preprocessing.py            # Cleaning, feature engineering, pipeline
│   ├── generate_eda_plots.py            # Generates all EDA visualizations
│   ├── model_training.py                # Trains, evaluates, saves final model
│   └── prediction.py                    # Predicts outcome for a new student
│
├── app/
│   └── app.py                           # Streamlit prediction interface
│
├── models/
│   ├── model.pkl                        # Saved trained pipeline (best model)
│   └── results.json                     # Saved evaluation metrics
│
├── images/                              # Saved chart images (used in this README)
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Data Cleaning

Starting shape: **1,215 rows** (1,200 base records + 15 duplicated rows,
injected deliberately to demonstrate cleaning).

| Issue found | Decision | Why |
|---|---|---|
| 15 exact duplicate rows | Dropped | A duplicate means the same student record appears twice, which would let the model "see" that student's outcome more than once and bias training. |
| Missing `CGPA`, `Attendance_Percentage`, `Communication_Score` (~2–3% each) | Filled with column **median** | Median is robust to skew; with such a small missing fraction, imputing preserves more data than dropping rows. |
| Missing `Extracurricular_Activities` (~2%) | Filled with column **mode** | Standard simple approach for a small fraction of missing categorical values. |
| Impossible values (e.g. attendance > 100%, CGPA > 10) | Checked, **none found** | Confirmed via `.describe()` before deciding no correction was needed. |

Result after cleaning: **1,200 rows, 0 missing values, 0 duplicates.**

## Exploratory Data Analysis

All charts below are generated by `src/generate_eda_plots.py` from the
cleaned dataset.

**Placement distribution:** roughly balanced — 617 Not Placed vs. 583 Placed
after cleaning (~51% / 49%). No special class-imbalance handling was needed.

**CGPA vs Placement:** Placed students show a visibly higher median CGPA
than students who were not placed, with more spread on the lower end for
Not Placed students. This is the single strongest individual numeric
correlate of placement in the dataset (correlation ≈ 0.18).

**Backlogs vs Placement:** Backlogs show a negative correlation with
placement (≈ -0.12) — students with more backlogs were placed somewhat
less often, though the relationship is not very strong.

**Correlation heatmap:** No numeric feature shows a very strong (>0.3)
individual linear correlation with placement — consistent with placement in
this dataset being a combination of many weak-to-moderate factors plus
randomness, rather than being driven by one dominant variable. The
engineered features `Total_Experience` and `Project_Experience` correlate
strongly with their source columns (as expected, since they're built from
them), so they shouldn't both be over-interpreted as independent signals.

**Important caveat:** these are correlations, not causal relationships. A
higher CGPA being associated with placement does not by itself prove that
raising CGPA *causes* placement — other unmeasured factors likely influence
both.

## Feature Engineering

| Feature | Formula | Why it might help |
|---|---|---|
| `Total_Experience` | `Internships + (Previous_Internship == "Yes")` | Combines two related "prior experience" signals into one number |
| `Skill_Average` | `mean(Technical_Skill_Score, Communication_Score)` | A simple combined soft+hard skill signal |
| `Project_Experience` | `Projects + Internships` | Combines hands-on project and internship counts into one "practical experience" measure |

All three are built only from information known **before** a placement
decision would be made, so none of them introduce data leakage.

## Data Leakage — What We Checked

Data leakage happens when a model has access, directly or indirectly, to
information it wouldn't realistically have at prediction time — making
evaluation results look better than they actually are.

Checks performed in this project:
- `Student_ID` excluded from model inputs (it's an identifier, not a real feature).
- `Placement_Status` (the target) never included as an input feature.
- All engineered features use only pre-placement information.
- The `StandardScaler` and `OneHotEncoder` are placed inside an sklearn
  `Pipeline`, which is fit **only** on the training set — they never see the
  test set until prediction/evaluation time.

## Train-Test Split

- **80% training (960 rows) / 20% testing (240 rows)**
- `random_state=42` for reproducibility
- `stratify=y` used, so both sets keep a similar Placed/Not-Placed ratio

Training data is what the model learns patterns from. Testing data is held
out and only used to check how well those patterns generalize to students
the model has never seen — training and evaluating on the same data would
give an overly optimistic, misleading result.

## Machine Learning Models

**Logistic Regression** — a linear classification model that estimates the
probability of an outcome (Placed vs Not Placed) as a weighted combination of
input features. It's simple, fast, and a good baseline to compare more
complex models against.

**Random Forest** — an ensemble of many decision trees, each trained on a
random subset of data and features. A single decision tree can overfit to
noise, but averaging predictions across many different trees ("the wisdom of
the crowd") tends to generalize better.

Both models were trained inside a pipeline with `StandardScaler` (numeric
features) and `OneHotEncoder` (categorical features) applied consistently.

## Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.629 | 0.635 | 0.564 | 0.597 | 0.686 |
| **Random Forest (selected)** | 0.629 | 0.632 | 0.573 | **0.601** | 0.676 |

*(Exact values from `models/results.json`, test set of 240 students.)*

**Confusion Matrix — Random Forest** (rows = actual, columns = predicted):

|  | Predicted: Not Placed | Predicted: Placed |
|---|---|---|
| **Actual: Not Placed** | 84 (True Negative) | 39 (False Positive) |
| **Actual: Placed** | 50 (False Negative) | 67 (True Positive) |

- **True Positive:** model correctly predicted "Placed" for a student who was actually placed.
- **True Negative:** model correctly predicted "Not Placed" for a student who wasn't placed.
- **False Positive:** model predicted "Placed" but the student wasn't (a false alarm).
- **False Negative:** model predicted "Not Placed" but the student was placed (a missed case).

- **Precision** (0.632): of the students the model predicted would be Placed,
  63.2% actually were. Relevant if you care about not over-promising placement.
- **Recall** (0.573): of the students who were actually Placed, the model
  correctly identified 57.3% of them. Relevant if you care about not missing
  students who are likely to succeed.
- **F1-Score** (0.601): the harmonic mean of precision and recall — a single
  balanced number, useful when both false positives and false negatives matter.

We did **not** rely on accuracy alone: with classes roughly balanced (~51/49),
accuracy is reasonably meaningful here, but F1-Score was still used as the
primary selection metric since it accounts for both types of errors.

## Model Comparison

Logistic Regression and Random Forest performed **almost identically**
(F1-Score 0.597 vs 0.601 — a 0.004 difference). Random Forest was selected
as the final model purely because it had the marginally higher F1-Score, not
because it's inherently "better" here — with a difference this small, the
choice isn't strongly meaningful, and Logistic Regression would have been an
equally defensible choice.

The modest performance (~63% accuracy, well above the 50% random baseline
but far from "excellent") is expected and consistent with how the synthetic
data was generated: placement outcomes were deliberately given a meaningful
random noise component, similar to how real placement outcomes involve
factors beyond what's measured here (interview mood, specific company needs,
timing, luck). This limits how high performance can realistically go on this
dataset, regardless of which model is used.

## Feature Importance

For the selected Random Forest model, the top contributing features were:

1. `CGPA` (0.133)
2. `Aptitude_Score` (0.106)
3. `Skill_Average` (0.096)
4. `Technical_Skill_Score` (0.094)
5. `Attendance_Percentage` (0.094)

*(Full ranking and chart in `images/13_feature_importance.png`.)*

**Important:** feature importance reflects how useful a feature was for the
model's *predictions* on this synthetic dataset — it does not prove that a
feature *causes* placement in the real world.

## New Student Prediction

`src/prediction.py` provides a `predict_new_student()` function that takes a
dictionary of a student's details and returns a predicted status and a
model-estimated probability, e.g.:

```
Prediction: Placed
Estimated probability: 77.5%
```

The probability comes directly from the trained model
(`pipeline.predict_proba`) — it is never hard-coded. It should be read as a
model estimate based on patterns in synthetic training data, **not** a
guarantee of any real student's actual outcome.

## How to Run

```bash
git clone <your-repo-url>
cd student-placement-predictor
pip install -r requirements.txt
```

**Regenerate the dataset (optional — a copy is already included):**
```bash
python src/generate_dataset.py
```

**Run the full analysis notebook:**
```bash
jupyter notebook notebooks/student_placement_analysis.ipynb
```

**Regenerate EDA charts:**
```bash
python src/generate_eda_plots.py
```

**Retrain models:**
```bash
python src/model_training.py
```

**Run the optional Streamlit app:**
```bash
streamlit run app/app.py
```
This opens a local browser tab where you can enter a student's details and
click "Predict Placement" to see the result.

## Key Insights

- CGPA was the single strongest individual numeric correlate of placement,
  though even it was only weakly-to-moderately correlated (≈0.18) —
  placement here isn't driven by one dominant factor.
- Backlogs showed a mild negative association with placement (≈ -0.12).
- Logistic Regression and Random Forest performed almost identically,
  suggesting the relationships in this dataset are close to linear/additive
  rather than requiring complex non-linear modeling.
- Overall predictive performance (~63% accuracy) is modest by design — a
  reminder that real placement outcomes depend on more than what's captured
  in structured data like this.

## Limitations

- **Synthetic data:** all results reflect patterns in artificially generated
  data, not real student records — they should not be treated as real-world
  findings.
- **Modest dataset size** (1,200 rows) limits how confidently patterns
  generalize.
- **Correlation ≠ causation:** relationships found here are associations,
  not proof that any feature causes placement.
- **Model predictions are estimates, not guarantees**, for any individual
  student.
- No hyperparameter tuning or cross-validation was performed — models used
  reasonable default/simple configurations.

## Future Improvements

- Train on a larger, real-world (properly licensed and anonymized) dataset.
- Hyperparameter tuning (e.g. `GridSearchCV`) for both models.
- k-fold cross-validation for more robust performance estimates.
- Try additional models (e.g. Gradient Boosting, SVM).
- Deploy the Streamlit app publicly (e.g. Streamlit Community Cloud).
- Add SHAP-based explainability for individual predictions.

## Author

**[Your Name Here]**
[Your GitHub Profile Link] · [Your LinkedIn Profile Link]
