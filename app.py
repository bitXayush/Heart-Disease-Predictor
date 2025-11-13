# app.py - Simple Heart Disease Predictor (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- load model ----
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("heart_model.joblib")
    return model

model = load_artifacts()

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Predictor")
st.markdown("Enter patient details below and click **Predict**. .")

# ---- input widgets ----
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=63)
        sex = st.selectbox("Sex (0 = female, 1 = male)", options=[0, 1], index=1)
        cp = st.selectbox("Chest pain type (cp)\n0: typical angina, 1: atypical, 2: non-anginal, 3: asymptomatic",
                          options=[0,1,2,3], index=3)
        trestbps = st.number_input("Resting BP (trestbps)", min_value=50, max_value=250, value=145)
    with col2:
        chol = st.number_input("Cholesterol (chol)", min_value=50, max_value=600, value=233)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0,1], index=1)
        restecg = st.selectbox("Resting ECG (restecg) (0,1,2)", options=[0,1,2], index=2)
        thalach = st.number_input("Max heart rate achieved (thalach)", min_value=50, max_value=250, value=150)
    with col3:
        exang = st.selectbox("Exercise induced angina (exang)", options=[0,1], index=0)
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=2.3, step=0.1, format="%.2f")
        slope = st.selectbox("ST slope (0,1,2)", options=[0,1,2], index=2)
        ca = st.selectbox("Number of major vessels (ca) (0-3)", options=[0,1,2,3], index=0)
        thal = st.selectbox("Thal (1 = normal, 2 = fixed defect, 3 = reversible defect)", options=[1,2,3], index=2)

    submitted = st.form_submit_button("Predict")

# ---- prediction ----
if submitted:
    # Build dataframe in the exact order used to train
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # model prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # probability of class 1 (disease)

    # display
    st.subheader("Result")
    if pred == 1:
        st.error(f"Prediction: **High risk of heart disease** (class = 1).")
    else:
        st.success(f"Prediction: **Low risk / No heart disease** (class = 0).")

    st.write(f"Probability of heart disease: **{prob:.3f}** ({prob*100:.1f}%)")

    # progress bar
    progress_val = int(prob * 100)
    st.progress(progress_val)

    # show input for report/debugging
    with st.expander("Input values (for report/debug)"):
        st.table(input_df.T.rename(columns={0:"value"}))

st.markdown("---")
st.caption("Model: RandomForestClassifier. This is an educational demo and not a medical device.")
