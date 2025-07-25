import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode categorical variables
sex_encoded = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Correct feature order
input_data = pd.DataFrame([[
    pclass, age, sibsp, parch, fare, sex_encoded, embarked_Q, embarked_S
]], columns=[
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_encoded', 'Embarked_Q', 'Embarked_S'
])

# Make prediction
if st.button("Predict"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.success("🎉 This passenger would have survived!")
    else:
        st.error("💀 Unfortunately, this passenger would not have survived.")