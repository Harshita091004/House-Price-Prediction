import streamlit as st 
import pickle
import numpy as np

# Load model
model = pickle.load(open('modelbest.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction App")

st.write("Enter property details below 👇")

# Inputs (better UI)
house_age = st.slider("Avg Area House Age", 1, 10, 5)
income = st.slider("Avg Area Income", 20000, 100000, 50000)
rooms = st.slider("Rooms", 2, 8, 5)
bedrooms = st.slider("Bedrooms", 1, 6, 3)
population = st.slider("Population", 5000, 40000, 20000)

# Prediction
if st.button("Predict Price 💰"):
    
    data = np.array([[income, house_age, rooms, bedrooms, population]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    prediction = max(0, prediction[0])  # prevent negative
    
    st.success(f"🏡 Estimated Price: $ {prediction:,.2f}")
    
    # Extra insight (PRO FEATURE)
    if income > 70000:
        st.info("💡 High income area → price tends to increase")
    if rooms > 6:
        st.info("💡 More rooms → higher property value")
    
    