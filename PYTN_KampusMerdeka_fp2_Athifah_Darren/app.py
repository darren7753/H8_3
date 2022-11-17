import numpy as np
import pandas as pd
import streamlit as st
from pickle import load

model1 = load(open("PYTN_KampusMerdeka_fp2_Athifah_Darren/logistic_regression_model.json","rb"))
model2 = load(open("PYTN_KampusMerdeka_fp2_Athifah_Darren/support_vector_machine_model.json","rb"))
data_clean = pd.read_csv("PYTN_KampusMerdeka_fp2_Athifah_Darren/data_clean.csv")
features = pd.read_csv("PYTN_KampusMerdeka_fp2_Athifah_Darren/features.csv")

@st.cache

def predict(model,mintemp,maxtemp,rainfall,windgustspeed,humidity9am,humidity3pm,pressure9am,pressure3pm,temp3pm,location,windgustdir,winddir9am,winddir3pm,raintoday):
    inputs = pd.DataFrame(columns=features.columns)
    location_code = [0 for i in inputs.columns[pd.Series(inputs.columns).str.startswith("location")]]
    if location != "Adelaide":
        n = 0
        for i in inputs.columns[pd.Series(inputs.columns).str.startswith("location")]:
            if location in i:
                location_code[n] = 1
                break
            n += 1 

    windgustdir_code = [0 for i in inputs.columns[pd.Series(inputs.columns).str.startswith("windgustdir")]]
    if windgustdir != "E":
        n = 0
        for i in inputs.columns[pd.Series(inputs.columns).str.startswith("windgustdir")]:
            if windgustdir in i:
                windgustdir_code[n] = 1
                break
            n += 1 

    winddir9am_code = [0 for i in inputs.columns[pd.Series(inputs.columns).str.startswith("winddir9am")]]
    if winddir9am != "E":
        n = 0
        for i in inputs.columns[pd.Series(inputs.columns).str.startswith("winddir9am")]:
            if winddir9am in i:
                winddir9am_code[n] = 1
                break
            n += 1 

    winddir3pm_code = [0 for i in inputs.columns[pd.Series(inputs.columns).str.startswith("winddir3pm")]]
    if winddir3pm != "E":
        n = 0
        for i in inputs.columns[pd.Series(inputs.columns).str.startswith("winddir3pm")]:
            if winddir3pm in i:
                winddir3pm_code[n] = 1
                break
            n += 1 

    if raintoday == "Yes":
        raintoday_code = [1]
    elif raintoday == "No":
        raintoday_code = [0]

    inputs = pd.DataFrame(
        data=[[mintemp,maxtemp,rainfall,windgustspeed,humidity9am,humidity3pm,pressure9am,pressure3pm,temp3pm] + location_code + windgustdir_code + winddir9am_code + winddir3pm_code + raintoday_code],
        columns=features.columns
    )

    pred= model.predict(inputs)[0]

    if pred == "Yes":
        return f"It will be raining tomorrow"

    elif pred == "No":
        return f"It won't be raining tomorrow"

st.markdown("<h1 style='text-align: center;'>Rain Prediction in Australia</h1>",unsafe_allow_html=True)
st.subheader("Choose the model:")
model = st.selectbox("Model",["Logistic Regression - Accuracy 76%","Support Vector Machine - Accuracy 80%"])
st.subheader("Input the required data:")

mintemp = st.number_input("MinTemp",value=10.00)
maxtemp = st.number_input("MaxTemp",value=20.00)
rainfall = st.number_input("Rainfall",value=0.25)
windgustspeed = st.number_input("WindGustSpeed",value=35.00)
humidity9am = st.number_input("Humidity9am",value=65.00)
humidity3pm = st.number_input("Humidity3am",value=50.00)
pressure9am = st.number_input("Pressure9am",value=1015.00)
pressure3pm = st.number_input("Pressure3am",value=1015.00)
temp3pm = st.number_input("Temp3pm",value=20.00)
location = st.selectbox("Location",list(np.sort(data_clean["location"].unique())))
windgustdir = st.selectbox("Windgustdir",list(np.sort(data_clean["windgustdir"].unique())))
winddir9am = st.selectbox("Winddir9am",list(np.sort(data_clean["winddir9am"].unique())))
winddir3pm = st.selectbox("Winddir3pm",list(np.sort(data_clean["winddir3pm"].unique())))
raintoday = st.selectbox("RainToday",list(np.sort(data_clean["raintoday"].unique())))

if model == "Logistic Regression - Accuracy 76%":
    if st.button("Rain Prediction"):
        rain = predict(model1,mintemp,maxtemp,rainfall,windgustspeed,humidity9am,humidity3pm,pressure9am,pressure3pm,temp3pm,location,windgustdir,winddir9am,winddir3pm,raintoday)
        st.success(rain)
elif model == "Support Vector Machine - Accuracy 80%":
    if st.button("Rain Prediction"):
        rain = predict(model2,mintemp,maxtemp,rainfall,windgustspeed,humidity9am,humidity3pm,pressure9am,pressure3pm,temp3pm,location,windgustdir,winddir9am,winddir3pm,raintoday)
        st.success(rain)