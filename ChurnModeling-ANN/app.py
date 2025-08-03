import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import streamlit as st
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Join the script directory with the filenames to create absolute paths
model_path = os.path.join(script_dir, 'model.keras')
gender_encoder_path = os.path.join(script_dir, 'gender_labelencoder.pkl')
geo_encoder_path = os.path.join(script_dir, 'geo_onehotencoder.pkl')
scaler_path = os.path.join(script_dir, 'standardscaler.pkl')

# --- Load Models and Encoders ---
# Load all your files using the correct paths
try:
    model = load_model(model_path)
    with open(gender_encoder_path, 'rb') as file:
        gender_labelencoder = pickle.load(file)
    with open(geo_encoder_path, 'rb') as file:
        geo_onehotencoder = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        standardscaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model or preprocessor files: {e}")
    st.stop() # Stop the app if files can't be loaded

## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', geo_onehotencoder.categories_[0])
gender = st.selectbox('Gender', gender_labelencoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_labelencoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = geo_onehotencoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_onehotencoder.get_feature_names_out(['Geography']))

input_final = pd.concat([input_data, geo_encoded_df], axis=1)


input_scaled=standardscaler.transform(input_final)

prediction=model.predict(input_scaled)
prob=prediction[0][0]
st.write(prob)
if prob>0.5:
    st.write('Customer Likely to Churn')
else:

    st.write('Customer not likely to churn')

