import streamlit as st
import pandas as pd
import joblib
import os

from src.data_preprocessing import preprocess_data
from src.evaluation import evaluate_model
from src.prediction import make_prediction
from src.visualization import (
    plot_hours_studied,
    plot_attendance,
    plot_target_distribution,
    plot_participation_distribution,
    plot_heatmap,
    plot_predictions_vs_actual,
    plot_residuals
)

st.set_page_config(
    page_title='Student Performance Regressor',
    layout='wide'
)

st.title('🎓 Student Performance Stacking Regressor Dashboard')
st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

target_data_path = None
if os.path.exists(DATA_DIR):
    for file in os.listdir(DATA_DIR):
        if 'student' in file.lower() and file.endswith('.csv'):
            target_data_path = os.path.join(DATA_DIR, file)
            break

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not target_data_path or not os.path.exists(MODEL_PATH):
    st.error("⚠️ Operational parameters or serialized model metadata missing!")
    st.info("Please drop your dataset into the data folder and run `python -m src.model_training` to train the engine.")
    st.stop()
else:
    df = load_data(target_data_path)
    model = load_model(MODEL_PATH)

st.sidebar.header('🎯 Pipeline Navigation')
page = st.sidebar.radio(
    'Go To:',
    ['Dataset Summary', 'Exploratory Data Analysis', 'Model Performance Evaluation', 'Interactive Performance Estimator']
)

if page == 'Dataset Summary':
    st.header('🗃️ Dataset Profile Insights')
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Matrix Sizing')
        st.info(f"Student Records: **{df.shape[0]}** | Measured Attributes: **{df.shape[1]}**")
    with col2:
        st.subheader('Missing Value Integrity Check')
        st.write(df.isnull().sum())

    st.subheader('Descriptive Statistics Metrics')
    st.dataframe(df.describe(), use_container_width=True)

elif page == 'Exploratory Data Analysis':
    st.header('📊 Interactive Feature Visualizations')
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_hours_studied(df), use_container_width=True)
        st.plotly_chart(plot_target_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_attendance(df), use_container_width=True)
        st.plotly_chart(plot_participation_distribution(df), use_container_width=True)
        
    st.markdown("---")
    st.plotly_chart(plot_heatmap(df), use_container_width=True)

elif page == 'Model Performance Evaluation':
    st.header('⚡ Regression Performance Validation')

    _, X_test, _, y_test, _ = preprocess_data(df)
    metrics, y_test_vec, y_pred_vec = evaluate_model(model, X_test, y_test)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📊 R-squared ($R^2$ Variance Score)", value=f"{round(metrics['r2'], 4)}")
    with col2:
        st.metric(label="🎯 Mean Absolute Error (MAE)", value=f"{round(metrics['mae'], 2)} Marks")
    with col3:
        st.metric(label="📉 Root Mean Squared Error (RMSE)", value=f"{round(metrics['rmse'], 2)} Marks")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_predictions_vs_actual(y_test_vec, y_pred_vec), use_container_width=True)
    with col2:
        st.plotly_chart(plot_residuals(y_test_vec, y_pred_vec), use_container_width=True)

elif page == 'Interactive Performance Estimator':
    st.header('🔮 Real-Time Student Score Forecasting')
    st.write("Modify academic inputs to forecast the student's expected continuous total score metric.")

    col1, col2 = st.columns(2)
    with col1:
        weekly_self_study_hours = st.slider('Weekly Self-Study Hours', 0, 80, 15)
        attendance_percentage = st.slider('Class Attendance Percentage (%)', 0, 100, 85)
    with col2:
        class_participation = st.slider('Class Participation Metric Score', 0, 100, 70)

    st.markdown("---")
    if st.button('Run Stacking Inference Engine', use_container_width=True):
        predicted_score = make_prediction(
            model, weekly_self_study_hours, attendance_percentage, class_participation
        )
        
        st.subheader('Estimated Performance Outcome')
        st.success(f"🎯 Predicted Student Total Score: **{round(predicted_score, 2)} / 100.0**")