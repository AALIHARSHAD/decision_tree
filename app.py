import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🧠 Customer Segmentation using K-Means")
st.write("Upload a customer dataset and perform K-Means clustering.")

# ----------------------------------
# File upload
# ----------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to continue")
    st.stop()

# ----------------------------------
# Load data
# ----------------------------------
df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.write("Columns:", df.columns.tolist())

# ----------------------------------
# Data Cleaning
# ----------------------------------
st.subheader("🧹 Data Cleaning")

# Drop ID if present
if "ID" in df.columns:
    df.drop("ID", axis=1, inplace=True)

# Separate numeric & categorical columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes
