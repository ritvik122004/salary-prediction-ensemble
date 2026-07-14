import streamlit as st

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Salary Prediction using Ensemble Learning")

st.markdown("""
This application predicts the salary of an employee using an
**Extra Trees Regressor** trained on employee demographic,
education and job-related information.
""")

st.write("Fill in the employee details below.")

with st.sidebar:
    st.title("📌 About Project")

    st.info(
        "This application predicts employee salary using an Extra Trees Regressor model."
    )

    st.markdown("### 📊 Model Details")
    st.write("- Algorithm: Extra Trees Regressor")
    st.write("- R² Score: 0.916")
    st.write("- Hyperparameter Tuned: ✅")

    st.markdown("---")
    st.markdown("👨‍💻 **Developer:** Ritvik Singh")
    

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_prediction_model (1).pkl")
model_columns = joblib.load("model_columns.pkl")

data = pd.read_csv("Salary Data.csv")


age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

gender = st.selectbox(
    "Gender",
    sorted(data["Gender"].dropna().unique())
)

education = st.selectbox(
    "Education Level",
    sorted(data["Education Level"].dropna().unique())
)

job_title = st.selectbox(
    "Job Title",
    sorted(data["Job Title"].dropna().unique())
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=2.0
)


if st.button("Predict Salary"):

    new_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job_title],
        "Years of Experience": [experience]
    })

    education_order = {
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    }

    new_data["Education Level"] = new_data["Education Level"].map(education_order)

    new_data = pd.get_dummies(new_data, drop_first=True)

    new_data = new_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(new_data)

    
    
    try:
        prediction = model.predict(new_data)

        st.metric(
        "Predicted Salary",
        f"₹ {prediction[0]:,.2f}"
        )

    except Exception as e:
        st.error(str(e))
