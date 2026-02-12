import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Student Health ML", layout="wide")

# Helper to load models
@st.cache_resource
def load_resources():
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le_target = pickle.load(open('le_target.pkl', 'rb'))
    model_names = ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
    models = {name: pickle.load(open(f"{name}.pkl", 'rb')) for name in model_names}
    return scaler, le_target, models

scaler, le_target, models = load_resources()

st.title("🏥 Student Health Risk Analysis")

# Requirement (a): Dataset Upload
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", test_data.head())

    # Requirement (b): Model Selection Dropdown
    st.sidebar.header("2. Model Settings")
    model_choice = st.sidebar.selectbox("Choose ML Model", 
                                        ["Logistic Regression", "Decision Tree", "kNN", 
                                         "Naive Bayes", "Random Forest", "XGBoost"])
    
    model_key = model_choice.lower().replace(" ", "_")
    selected_model = models[model_key]

    if st.sidebar.button("Run Evaluation"):
        # Preprocessing test data
        # Note: In a real app, you'd apply the same encoding as training
        # For demo purposes, we assume the CSV is already pre-processed for features
        # or we exclude non-numeric columns for calculation
        X_test = test_data.select_dtypes(include=[np.number])
        if 'Health_Risk_Level' in test_data.columns:
            y_test = le_target.transform(test_data['Health_Risk_Level'])
            X_test = X_test.drop(columns=['Student_ID'], errors='ignore')
            
            # Predict
            X_test_scaled = scaler.transform(X_test)
            y_pred = selected_model.predict(X_test_scaled)

            # Requirement (c): Display Evaluation Metrics
            st.subheader(f"📊 Evaluation Metrics: {model_choice}")
            report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.table(df_report)

            # Requirement (d): Confusion Matrix
            st.subheader("🖼️ Confusion Matrix")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le_target.classes_, yticklabels=le_target.classes_)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
else:
    st.info("Please upload a test dataset (CSV) in the sidebar to begin.")