import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np


file_path = "G:/ML Project 1/StudentPerformanceFactors.csv"
df = pd.read_csv(file_path)

X = df[['Hours_Studied', 'Attendance', 'Previous_Scores']]
y = df['Exam_Score'].apply(lambda score: 1 if score >= 65 else 0)


model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "Student_Performance_Analyser.joblib")



model = joblib.load("Student_Performance_Analyser.joblib")  

print(type(model))


st.title("Student Performance Prediction")
study_hours = st.number_input("Hours_Studied", min_value=0.0, max_value=44.0, value=23.0)
attendance = st.number_input("Attendance", min_value=0, max_value=100, value=84)
past_grade = st.number_input("Previous_Scores", min_value=0, max_value=100, value=73)



input_data = pd.DataFrame([[study_hours, attendance, past_grade]],
                          columns=['Hours_Studied', 'Attendance', 'Previous_Scores'])

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("The student is predicted to Pass.")
    else:
        st.error("The student is predicted to Fail.")
