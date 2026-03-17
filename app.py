import streamlit as st
import pandas as pd
import joblib

model = joblib.load('recovery_model.pkl')

st.title("🛡️ VitalSense: Daily Recovery Predictor")
st.write("Predict your body's readiness based on yesterday's activity.")

st.sidebar.header("Yesterday's Stats")
steps = st.sidebar.slider("Steps Taken", 0, 20000, 8000)
active_mins = st.sidebar.slider("Intense Exercise (Mins)", 0, 120, 30)
sedentary_mins = st.sidebar.slider("Sitting Time (Mins)", 0, 1000, 480)
calories = st.sidebar.slider("Calories Burned", 1000, 4000, 2200)

# Prediction Logic
input_data = pd.DataFrame([[steps, active_mins, sedentary_mins, calories]], 
                          columns=['TotalSteps', 'VeryActiveMinutes', 'SedentaryMinutes', 'Calories'])

prediction = model.predict(input_data)[0]

st.subheader(f"Your Predicted Recovery Score: {round(prediction, 1)}%")

if prediction > 80:
    st.success("🚀 **Peak Condition:** Your body is well-rested and ready for high-intensity training.")
elif prediction > 50:
    st.warning("⚖️ **Moderate Recovery:** You're doing okay, but don't push for a personal record today.")
else:
    st.error("🛑 **Low Recovery:** Your metrics suggest high fatigue. Prioritize sleep and hydration.")

st.subheader("💡 Personalized Tips for Tomorrow")

tips = []

if steps < 5000:
    tips.append("🚶 **Move More:** Your steps are low. Aim for at least 8,000 steps today to boost cardiovascular circulation.")
elif steps > 15000:
    tips.append("🧘 **Active Recovery:** You moved a lot yesterday! Try some light stretching or yoga to avoid burnout.")

if sedentary_mins > 600:
    tips.append("🪑 **Break the Sedentary Cycle:** You spent over 10 hours sitting. Set a timer to stand up and move every 60 minutes.")

if active_mins < 15:
    tips.append("⚡ **Heart Rate Spike:** Try to get at least 15 minutes of elevated heart rate to improve your long-term Resting HR.")

for tip in tips:
    st.info(tip)