# Libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pickle import load
from PIL import Image

# Load
model1 = load(open("PYTN_KampusMerdeka_fp2_Athifah_Darren/logistic_regression_model.json","rb"))
model2 = load(open("PYTN_KampusMerdeka_fp2_Athifah_Darren/support_vector_machine_model.json","rb"))
data_clean = pd.read_csv("PYTN_KampusMerdeka_fp2_Athifah_Darren/data_clean.csv")
features = pd.read_csv("PYTN_KampusMerdeka_fp2_Athifah_Darren/features.csv")
cover_img = Image.open("PYTN_KampusMerdeka_fp2_Athifah_Darren/dataset-cover.jpg")

@st.cache

# Function for Predicting
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

    pred = model.predict(inputs)[0]
    if model == model1:
        if pred == "Yes":
            proba_not_rain = model.predict_proba(inputs)[0,0]
            proba_rain = model.predict_proba(inputs)[0,1]
            return ["It's likely to rain tomorrow",proba_not_rain,proba_rain]
        else:
            proba_not_rain = model.predict_proba(inputs)[0,0]
            proba_rain = model.predict_proba(inputs)[0,1]
            return ["It's unlikely to rain tomorrow",proba_not_rain,proba_rain]
    else:
        if pred == "Yes":
            return "It's likely to rain tomorrow"
        else:
            return "It's unlikely to rain tomorrow"

# Title
st.markdown("<h1 style='text-align: center;'>Rain Prediction in Australia</h1>",unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Made by <b><a href='https://www.linkedin.com/in/athifahrh/'>'Athifah Radhiyah Habibilah</a></b> & <b><a href='https://www.linkedin.com/in/mathewdarren/'>Mathew Darren Kusuma</a></b></p>",
    unsafe_allow_html=True
)
st.image(cover_img)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Model
st.header("Choose Model")
model = st.selectbox("Model",["Logistic Regression","Support Vector Machine"],label_visibility="collapsed")
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Input
st.header("Input Data")
col1,col2 = st.columns(2)

with col1:
    mintemp = st.number_input("MinTemp",value=10.00)
    maxtemp = st.number_input("MaxTemp",value=20.00)
    rainfall = st.number_input("Rainfall",value=0.25)
    windgustspeed = st.number_input("WindGustSpeed",value=35.00)
    humidity9am = st.number_input("Humidity9am",value=65.00)
    humidity3pm = st.number_input("Humidity3am",value=50.00)
    pressure9am = st.number_input("Pressure9am",value=1015.00)

with col2:
    pressure3pm = st.number_input("Pressure3am",value=1015.00)
    temp3pm = st.number_input("Temp3pm",value=20.00)
    location = st.selectbox("Location",list(np.sort(data_clean["location"].unique())))
    windgustdir = st.selectbox("Windgustdir",list(np.sort(data_clean["windgustdir"].unique())))
    winddir9am = st.selectbox("Winddir9am",list(np.sort(data_clean["winddir9am"].unique())))
    winddir3pm = st.selectbox("Winddir3pm",list(np.sort(data_clean["winddir3pm"].unique())))
    raintoday = st.selectbox("RainToday",list(np.sort(data_clean["raintoday"].unique())))

st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Prediction
st.header("Prediction")

if model == "Logistic Regression":
    if st.button("Click here to predict"):        
        rain = predict(model1,mintemp,maxtemp,rainfall,windgustspeed,humidity9am,humidity3pm,pressure9am,pressure3pm,temp3pm,location,windgustdir,winddir9am,winddir3pm,raintoday)

        fig = go.Figure(go.Bar(
            x=[rain[1] * 100,rain[2] * 100],
            y=["Not Rain","Rain"],
            orientation="h",
            marker=dict(color=["rgb(26,118,255)" if prob == np.max(rain[1:]) else "rgb(55,83,109)" for prob in rain[1:]]),
        ))
        fig.update_traces(hovertemplate=None)
        fig.update_layout(
            title="Class Probabilities",
            title_x=0.5,
            title_font_size=25,
            xaxis=dict(
                tickfont=dict(size=15),
                ticksuffix="%",
                showgrid=False,
                ticks="outside",
                tickcolor="white",
                ticklen=10
            ),
            yaxis=dict(
                tickfont=dict(size=15),
                showgrid=False,
                ticks="outside",
                tickcolor="white",
                ticklen=10)
            )
        st.plotly_chart(fig,use_container_width=True)

        st.success(rain[0])
else:
    if st.button("Click here to predict"):
        rain = predict(model2,mintemp,maxtemp,rainfall,windgustspeed,humidity9am,humidity3pm,pressure9am,pressure3pm,temp3pm,location,windgustdir,winddir9am,winddir3pm,raintoday)
        st.info("The Class Probabilities plot is unavailable for the Support Vector Machine model")
        st.success(rain)