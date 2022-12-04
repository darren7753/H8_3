# Libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pickle import load
from PIL import Image

# Load
rf = load(open("PYTN_KampusMerdeka_fp3_Athifah_Darren/random_forest_smote_enn_model.json","rb"))
gb = load(open("PYTN_KampusMerdeka_fp3_Athifah_Darren/gradient_boosting_smote_enn_model.json","rb"))
df = pd.read_csv("PYTN_KampusMerdeka_fp3_Athifah_Darren/heart_failure_clinical_records_dataset.csv")
cover_img = Image.open("PYTN_KampusMerdeka_fp3_Athifah_Darren/dataset-cover.png")

# Title
st.markdown("<h1 style='text-align: center;'>Heart Failure Prediction</h1>",unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Made by <b><a href='https://www.linkedin.com/in/athifahrh/'>'Athifah Radhiyah Habibilah</a></b> & <b><a href='https://www.linkedin.com/in/mathewdarren/'>Mathew Darren Kusuma</a></b></p>",
    unsafe_allow_html=True
)
st.image(cover_img)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Model
st.header("Choose Model")
choose_model = st.selectbox("Model",["Random Forest","Gradient Boosting"],label_visibility="collapsed")
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Input
st.header("Input Data")
col1,col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)",value=50)
    anaemia = st.selectbox("Anaemia (Boolean)",["Yes","No"])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)",value=250)
    diabetes = st.selectbox("Diabetes (Boolean)",["Yes","No"])
    ejection_fraction = st.number_input("Ejection Fraction (Percentage)",value=40)
    high_blood_pressure = st.selectbox("High Blood Pressure (Boolean)",["Yes","No"])

with col2:
    platelets = st.number_input("Platelets (kiloplatelets/mL)",value=250_000.00)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)",value=1.15)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)",value=140)
    sex = st.selectbox("Sex (Boolean)",["Male","Female"])
    smoking = st.selectbox("Smoking (Boolean)",["Yes","No"])
    time = st.number_input("Time (Days)",value=80)

st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Prediction
st.header("Prediction")

inputs = [
    age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,
    platelets,serum_creatinine,serum_sodium,sex,smoking,time
]

for index,input_ in enumerate(inputs):
    if (input_ == "Yes") or (input_ == "Male"):
        inputs[index] = 1
    elif (input_ == "No") or (input_ == "Female"):
        inputs[index] = 0

if choose_model == "Random Forest":
    model = rf
else:
    model = gb

predict = model.predict(pd.DataFrame([inputs],columns=df.columns[:-1]))[0]
proba = model.predict_proba(pd.DataFrame([inputs],columns=df.columns[:-1]))[0]

if st.button("Click here to predict"):
    fig = go.Figure(go.Bar(
        x=[proba[0] * 100,proba[1] * 100],
        y=["Dead","Survive"],
        orientation="h",
        marker=dict(color=["rgb(26,118,255)" if prob == np.max(proba) else "rgb(55,83,109)" for prob in proba]),
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

    if predict == "Survived":
        st.success("This patient is likely going to survive")
    else:
        st.success("This patient is likely going to die")