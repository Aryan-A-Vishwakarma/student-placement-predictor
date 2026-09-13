"""
app.py

A simple Streamlit app that lets a user enter a student's details and get
a placement prediction + probability from the trained model.

Run locally with:
    streamlit run app/app.py
(run this command from the project root folder)
"""

import sys
import os

# Allow importing from src/ regardless of where streamlit is launched from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from prediction import predict_new_student

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓")

st.title("🎓 Student Placement Predictor")
st.write(
    "Enter a student's academic and extracurricular details to estimate "
    "their likelihood of campus placement. "
    "**Note:** this model was trained on synthetic data for demonstration "
    "purposes — treat predictions as illustrative, not authoritative."
)

st.header("Student Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=30, value=21)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
    attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0, 0.5)
    study_hours = st.slider("Study Hours Per Day", 0.0, 12.0, 3.0, 0.5)
    backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=10, value=0)
    internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1)

with col2:
    projects = st.number_input("Number of Projects", min_value=0, max_value=10, value=2)
    certifications = st.number_input("Number of Certifications", min_value=0, max_value=10, value=1)
    communication_score = st.slider("Communication Score", 0.0, 100.0, 65.0, 1.0)
    technical_skill_score = st.slider("Technical Skill Score", 0.0, 100.0, 65.0, 1.0)
    aptitude_score = st.slider("Aptitude Score", 0.0, 100.0, 60.0, 1.0)
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    previous_internship = st.selectbox("Previous Internship Experience", ["Yes", "No"])

if st.button("Predict Placement", type="primary"):
    student_info = {
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
        "Gender": gender,
        "Extracurricular_Activities": extracurricular,
        "Previous_Internship": previous_internship,
    }

    result = predict_new_student(student_info)

    st.subheader("Prediction Result")
    if result["prediction"] == "Placed":
        st.success(f"Predicted Status: **{result['prediction']}**")
    else:
        st.warning(f"Predicted Status: **{result['prediction']}**")

    st.metric("Estimated Placement Probability", f"{result['probability'] * 100:.1f}%")
    st.caption(
        "This probability is a model estimate based on patterns learned from "
        "synthetic training data — it is not a guarantee of a real outcome."
    )
