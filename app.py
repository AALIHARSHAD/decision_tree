import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Decision Tree App", layout="wide")

st.title("🌳 Decision Tree Classification")
st.write("Social Network Ads Dataset")

# ----------------------------------
# Load dataset
# ----------------------------------
df = pd.read_csv("Social_Network_Ads.csv")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.write("Columns:", df.columns.tolist())

# ----------------------------------
# Drop ID column
# ----------------------------------
if "User ID" in df.columns:
    df.drop("User ID", axis=1, inplace=True)

# ----------------------------------
# Encode categorical columns
# ----------------------------------
if "Gender" in df.columns:
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

# ----------------------------------
# Features & Target
# ----------------------------------
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# ----------------------------------
# Feature scaling
# ----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------
# Train-test split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# ----------------------------------
# Train Decision Tree
# ----------------------------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# Prediction
# ----------------------------------
y_pred = model.predict(X_test)

# ----------------------------------
# Accuracy
# ----------------------------------
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.metric("Accuracy", round(accuracy, 3))

# ----------------------------------
# Confusion Matrix
# ----------------------------------
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

# ----------------------------------
# Classification Report
# ----------------------------------
st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))

# ----------------------------------
# Decision Tree Visualization
# ----------------------------------
st.subheader("🌳 Decision Tree Structure")

fig2, ax2 = plt.subplots(figsize=(14, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Purchased", "Purchased"],
    filled=True,
    ax=ax2
)
st.pyplot(fig2)

