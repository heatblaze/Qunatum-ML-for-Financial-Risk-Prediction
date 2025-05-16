import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
import pandas as pd
import numpy as np
from src.data_preprocessing import load_and_preprocess_data
from src.classical_baseline import train_classical_model, evaluate_model
from src.quantum_reservoir import quantum_feature_map, train_quantum_classifier

st.set_page_config(page_title="Quantum Financial Risk Predictor", layout="wide")
st.title("💡 Quantum Machine Learning for Financial Risk Prediction")

# Sidebar - Data upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    X_train, X_test, y_train, y_test = load_and_preprocess_data(uploaded_file)
    st.success("Data loaded and preprocessed!")

    # Classical Model
    st.header("Classical Baseline")
    clf = train_classical_model(X_train, y_train)
    report = evaluate_model(clf, X_test, y_test)
    st.write("**Classification Report (Classical):**")
    st.json(report)

    # Quantum Model
    st.header("Quantum Reservoir Model")
    n_qubits = st.slider("Number of Qubits", 2, 4, 3)
    st.info("Extracting quantum features (this may take a minute)...")
    X_train_q = quantum_feature_map(X_train[:30], n_qubits)  # Limit for demo speed
    X_test_q = quantum_feature_map(X_test[:10], n_qubits)
    q_clf = train_quantum_classifier(X_train_q, y_train[:30])
    q_pred = q_clf.predict(X_test_q)
    from sklearn.metrics import classification_report
    q_report = classification_report(y_test[:10], q_pred, output_dict=True)
    st.write("**Classification Report (Quantum):**")
    st.json(q_report)

    # Predict on new data
    st.header("Try a Prediction")
    user_input = st.text_input("Enter comma-separated feature values:")
    if user_input:
        features = np.array([float(x) for x in user_input.split(",")])
        features = features.reshape(1, -1)
        # Classical prediction
        class_pred = clf.predict(features)
        # Quantum prediction
        q_feat = quantum_feature_map(features, n_qubits)
        q_pred = q_clf.predict(q_feat)
        st.write(f"Classical Prediction: {'Risk' if class_pred[0] else 'No Risk'}")
        st.write(f"Quantum Prediction: {'Risk' if q_pred[0] else 'No Risk'}")
else:
    st.info("Please upload a CSV file to get started.")

st.caption("Powered by Qiskit, Scikit-learn, and Streamlit.")
