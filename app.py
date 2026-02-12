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
    model_keys = ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
    models = {k: pickle.load(open(f"{k}.pkl", 'rb')) for k in model_keys}
    return scaler, le_target, models

# Verify files exist before loading
try:
    scaler, le_target, models = load_resources()
except FileNotFoundError:
    st.error("Model files not found. Please run model_training.py first!")
    st.stop()

st.title("🎓 Student Health Risk Classifier")

# Sidebar
st.sidebar.header("Step 1: Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV for evaluation", type="csv")

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)
    st.write("### Test Dataset Overview", df_test.head())

    st.sidebar.header("Step 2: Model Settings")
    choice = st.sidebar.selectbox("Select Model", 
                                  ["Logistic Regression", "Decision Tree", "kNN", 
                                   "Naive Bayes", "Random Forest", "XGBoost"])
    
    model_key = choice.lower().replace(" ", "_")
    selected_model = models[model_key]

    if st.sidebar.button("Execute Model Evaluation"):
        # Processing inputs
        X_test = df_test.select_dtypes(include=[np.number]).drop(columns=['Student_ID'], errors='ignore')
        if 'Health_Risk_Level' in df_test.columns:
            y_test = le_target.transform(df_test['Health_Risk_Level'])
            X_test_scaled = scaler.transform(X_test)
            y_pred = selected_model.predict(X_test_scaled)

            # Display Metrics
            st.subheader(f"📊 {choice} Performance")
            report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
            st.table(pd.DataFrame(report).transpose())

            # Confusion Matrix
            st.subheader("🖼️ Confusion Matrix Visualization")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                        xticklabels=le_target.classes_, yticklabels=le_target.classes_)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            st.pyplot(fig)
else:
    st.info("👈 Upload a test CSV to start evaluation.")