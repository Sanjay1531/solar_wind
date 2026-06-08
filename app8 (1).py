import streamlit as st
import numpy as np
import joblib

# Load models and scaler
solar_model = joblib.load("solar_model_v1.pkl")
wind_model = joblib.load("wind_model_v1.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🌞💨 Solar & Wind Power Prediction App")
st.write("Enter environmental conditions to predict solar and wind power output.")

# User Inputs
solar_radiation = st.number_input("☀️ Solar Radiation (W/m²)", min_value=0.0, max_value=1200.0, value=500.0)
wind_speed = st.number_input("🌬 Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=10.0)
temperature = st.number_input("🌡 Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
pressure = st.number_input("🌪 Air Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)

# Prediction Function
def predict_power(solar_radiation, wind_speed, temperature, humidity, pressure):
    input_data = np.array([[solar_radiation, wind_speed, temperature, humidity, pressure]])
    scaled_input = scaler.transform(input_data)  # Scale input
    solar_output = solar_model.predict(scaled_input)[0]
    wind_output = wind_model.predict(scaled_input)[0]
    return solar_output, wind_output

# Predict Button
if st.button("⚡ Predict Power Output"):
    solar_power, wind_power = predict_power(solar_radiation, wind_speed, temperature, humidity, pressure)
    st.success(f"🔆 Predicted Solar Power Output: {solar_power:.2f} kW")
    st.success(f"🌪 Predicted Wind Power Output: {wind_power:.2f} kW")

# Footer
st.write("Developed by sanjay | Powered by Machine Learning 🚀")
