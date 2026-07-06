import streamlit as st
import joblib
import numpy as np

st.title("AI Learning Recommendation System")

st.write("Enter student assessment details to predict performance")

# Load model and scaler
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Input fields
weight = st.number_input("Assessment Weight", value=10)
date = st.number_input("Assessment Date", value=20)
assessment_type = st.number_input("Assessment Type", value=1)
code_module = st.number_input("Module Code", value=3)
submission_delay = st.number_input("Submission Delay", value=2)
student_avg_score = st.number_input("Student Average Score", value=65)
assessment_difficulty = st.number_input("Assessment Difficulty", value=10)

if st.button("Predict Student Performance"):

    data = np.array([[weight, date, assessment_type, code_module,
                      submission_delay, student_avg_score, assessment_difficulty]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Weak Student")
        st.write("Recommendation: Provide extra practice materials and revision sessions.")
    else:
        st.success("Strong Student")
        st.write("Recommendation: Provide advanced learning resources.")