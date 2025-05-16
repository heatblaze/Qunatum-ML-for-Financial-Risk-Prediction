# ⚛️ Quantum Financial Risk Predictor

A futuristic, interactive dashboard for predicting financial credit risk using both **classical** and **quantum machine learning** models. Upload your own dataset, analyze risk, and get detailed, explainable predictions-all within a visually striking, dark-themed web app.

---

## 🚀 Features

- **Dual Modeling:** Compare predictions from a classical Random Forest and a Quantum Reservoir Computing (QRC) model.
- **Modern UI:** Neon-accented dark theme, gradient headers, glassmorphism cards, and interactive Plotly charts.
- **Easy Data Upload:** Upload your own CSV file with financial data.
- **Robust Preprocessing:** Automatic one-hot encoding and scaling.
- **Explainable AI:** Feature importance charts, detailed markdown explanations, and quantum feature contribution graphs.
- **Ready for Real-World Data:** Handles categorical and numerical features, and displays clear error messages.

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/heatblaze/Qunatum-ML-for-Financial-Risk-Prediction
    ```
    ```bash
    cd quantum-financial-risk-predictor
    ```

2. **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **(Optional) Set up the dark theme:**  
   The `.streamlit/config.toml` is already included for a futuristic look.

5. **Run the app:**
    ```bash
    streamlit run app/dashboard.py
    ```

---

## 📈 Usage

1. **Upload your dataset** (CSV with a `target` column for risk label).
2. **Review model training and evaluation** for both classical and quantum models.
3. **Enter new applicant data** for prediction (app will show the required feature order).
4. **Explore detailed explanations and interactive charts** for each prediction.

---

## 💡 How It Works

- **Classical Model:** Uses Random Forest to predict risk based on engineered features.
- **Quantum Model:** Uses Quantum Reservoir Computing to encode data into quantum states, extract complex patterns, and predict risk via a classical readout.
- **Explainability:** The app displays feature importance (bar chart) for the classical model and a dark-themed line graph for quantum feature contributions, with detailed markdown explanations for both.

---

## 🧑‍💻 Technologies Used

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Qiskit](https://qiskit.org/)
- [Plotly](https://plotly.com/python/)
- [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)

---

## 📸 Screenshot

*(Add a screenshot of your app here!)*

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue first to discuss any changes.

---

## 📬 Contact

For questions or collaboration, open an issue or contact [your email/LinkedIn].

---
