import streamlit as st
import pickle
import numpy as np
import os

st.title("🚗 Car Evaluation Prediction (Decision Tree)")

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "decision_tree_car_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found.\n\nPlease run your notebook to train and save the model first.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Car Features")

buying = st.selectbox("Buying Price", ["low","med","high","vhigh"])
maint = st.selectbox("Maintenance Cost", ["low","med","high","vhigh"])
doors = st.selectbox("Number of Doors", ["2","3","4","5more"])
persons = st.selectbox("Seating Capacity", ["2","4","more"])
lug_boot = st.selectbox("Luggage Boot Size", ["small","med","big"])
safety = st.selectbox("Safety", ["low","med","high"])

# -----------------------------
# Encode Inputs (same as training)
# -----------------------------
mapping = {
    "low":0, "med":1, "high":2, "vhigh":3,
    "2":0, "3":1, "4":2, "5more":3,
    "small":0, "med":1, "big":2,
    "more":2
}

X = np.array([[
    mapping[buying],
    mapping[maint],
    mapping[doors],
    mapping[persons],
    mapping[lug_boot],
    mapping[safety]
]])

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    pred = model.predict(X)[0]

    class_mapping = {
        0: "Unacceptable",
        1: "Acceptable",
        2: "Good",
        3: "Very Good"
    }

    st.success(f"Prediction: ⭐ {class_mapping[pred]}")
