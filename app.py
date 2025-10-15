import streamlit as st
import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")

st.title("California House Price Predictor")
st.write("Enter the following features to predict the house price(in $100,000s):")

MedInc = st.number_input("Median Income (in $10,000s)", 0.0, 20.0, 5.0)
HouseAge = st.number_input("House Age (in years)", 1.0, 50.0, 20.0)
AveRooms = st.number_input("Average Rooms per Household", 1.0, 10.0, 5.0)
AveBedrms = st.number_input("Average Bedrooms per Household", 0.5, 5.0, 1.0)
Population = st.number_input("Population", 1.0, 5000.0, 500.0)
AveOccup = st.number_input("Average Occupants per Household", 1.0, 10.0, 3.0)
Latitude = st.number_input("Latitude", 32.0, 42.0, 35.0)
Longitude = st.number_input("Longitude", -125.0, -114.0, -120.0)

if st.button("Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Estimated House Price: ${prediction * 100000:.2f}")


