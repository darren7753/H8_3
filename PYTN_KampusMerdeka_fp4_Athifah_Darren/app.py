# Libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pickle import load
from PIL import Image

# Load
fa = load(open("PYTN_KampusMerdeka_fp4_Athifah_Darren/factor_analyze.pkl","rb"))
kmeans = load(open("PYTN_KampusMerdeka_fp4_Athifah_Darren/kmeans_clustering.pkl","rb"))
cover_img = Image.open("PYTN_KampusMerdeka_fp4_Athifah_Darren/dataset-cover.jpg")
df = pd.read_csv("PYTN_KampusMerdeka_fp4_Athifah_Darren/clean_dataset.csv")
components = pd.read_csv("PYTN_KampusMerdeka_fp4_Athifah_Darren/components.csv")

# Title
st.markdown("<h1 style='text-align: center;'>Credit Card Clustering</h1>",unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Made by <b><a href='https://www.linkedin.com/in/athifahrh/'>'Athifah Radhiyah Habibilah</a></b> & <b><a href='https://www.linkedin.com/in/mathewdarren/'>Mathew Darren Kusuma</a></b></p>",
    unsafe_allow_html=True
)
st.image(cover_img)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Input
st.header("Input Data")
col1,col2,col3 = st.columns(3)

with col1:
    balance = st.number_input("Balance",min_value=0.00,value=0.00)
    balance_frequency = st.number_input("Balance Frequency",min_value=0.00,max_value=1.00,value=0.00)
    purchases = st.number_input("Purchases",min_value=0.00,value=0.00)
    oneoff_purchases = st.number_input("One-Off Purchases",min_value=0.00,value=0.00)
    installments_purchases = st.number_input("Installments Purchases",min_value=0.00,value=0.00)
    cash_advance = st.number_input("Cash Advance",min_value=0.00,value=0.00)

with col2:
    purchases_frequency = st.number_input("Purchases Frequency",min_value=0.00,max_value=1.00,value=0.00)
    oneoff_purchases_frequency = st.number_input("One-Off Purchases Frequency",min_value=0.00,max_value=1.00,value=0.00)
    purchases_installments_frequency = st.number_input("Purchases Installments Frequency",min_value=0.00,max_value=1.00,value=0.00)
    cash_advance_frequency = st.number_input("Cash Advance Frequency",min_value=0.00,value=0.00)
    cash_advance_trx = st.number_input("Cash Advance Transactions",min_value=0.00,value=0.00)
    purchases_trx = st.number_input("Purchases Transactions",min_value=0.00,value=0.00)

with col3:
    credit_limit = st.number_input("Credit Limit",min_value=0.00,value=0.00)
    payments = st.number_input("Payments",min_value=0.00,value=0.00)
    minimum_payments = st.number_input("Minimum Payments",min_value=0.00,value=0.00)
    prc_full_payment = st.number_input("Percentage of Full Payments",min_value=0.00,value=0.00)
    tenure = st.number_input("Tenure",min_value=0.00,value=0.00)

st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# Factor Analysis
st.header("Dimensionality Reduction with Factor Analysis")

if balance_frequency == "frequently updated".title():
    balance_frequency = 1
else:
    balance_frequency = 0

if purchases_frequency == "frequently purchased".title():
    purchases_frequency = 1
else:
    purchases_frequency = 0

if oneoff_purchases_frequency == "frequently purchased".title():
    oneoff_purchases_frequency = 1
else:
    oneoff_purchases_frequency = 0

if purchases_installments_frequency == "frequently done".title():
    purchases_installments_frequency = 1
else:
    purchases_installments_frequency = 0

inputs = [
    balance,balance_frequency,purchases,oneoff_purchases,installments_purchases,cash_advance,
    purchases_frequency,oneoff_purchases_frequency,purchases_installments_frequency,cash_advance_frequency,
    cash_advance_trx,purchases_trx,credit_limit,payments,minimum_payments,prc_full_payment,tenure
]

new_values = pd.DataFrame(
    data=fa.transform(pd.DataFrame(data=[inputs],columns=df.columns)),
    columns=[f"Component {component + 1}" for component in range(5)],
)

components_and_new_values = pd.concat([new_values,components],axis=0).reset_index(drop=True)
components_and_new_values.index = ["Value"] + [f"Feature {var}" for var in range(1,7)]

st.info("The dimensions of the data is reduced into a subset of 5 dimensions",icon="ℹ️")
st.dataframe(components_and_new_values.T,width=1000)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;"/>""",unsafe_allow_html=True)

# K-Means Clustering
st.header("Clustering with K-Means")
st.info("The final cluster centers are computed as the mean for each feature within each final cluster",icon="ℹ️")

cluster_centers = pd.DataFrame(
    data=kmeans.cluster_centers_,
    columns=[f"Component {component + 1}" for component in range(5)],
    index=[f"Cluster {cluster}" for cluster in range(7)]
).T
st.dataframe(cluster_centers,width=1000)

predicted_cluster = kmeans.predict(new_values)[0]
distances = kmeans.transform(new_values)[0]

if st.button("Click here for clustering"):
    fig = go.Figure(go.Bar(
        x=[f"Cluster {cluster}" for cluster in range(7)],
        y=distances,
        marker=dict(color=["rgb(26,118,255)" if distance == np.min(distances) else "rgb(55,83,109)" for distance in distances])
    ))
    fig.update_layout(
        title="The Distances from Object to Each Cluster Center",
        title_x=0.5,
        title_font_size=25,
        xaxis=dict(
            tickfont=dict(size=15)
        ),
        yaxis=dict(
            tickfont=dict(size=15),
            showgrid=False,
            ticks="outside",
            tickcolor="white",
            ticklen=5
        )
    )
    st.plotly_chart(fig,use_container_width=True)

    st.success(f"This customer likely belongs to cluster {predicted_cluster}",icon="✅")