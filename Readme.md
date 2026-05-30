# Student Performance Stacking Regressor Dashboard 🎓📈

An end-to-end Machine Learning regression pipeline and interactive web application built with **Streamlit** and **Plotly** to predict a student's continuous `total_score` based on their academic habits and attendance records.

This project is built using a modular data science directory structure, separating processing logic, evaluation metrics, and the web interface layer.

---

## 🏢 Project Directory Layout

```text
student_performance_reg/
│
├── app.py                      # Main Streamlit Dashboard web interface
├── requirements.txt            # Project dependencies and packages
├── README.md                   # Project documentation
│
├── data/
│   └── student_performance.csv # Student academic dataset (Drop file here)
│
├── models/
│   └── model.pkl               # Serialized Stacking Regressor Pipeline artifact
│
└── src/
    ├── __init__.py             # Marks directory as a Python package
    ├── data_preprocessing.py   # Cleans data, drops IDs/grades, extracts target feature
    ├── model_training.py       # Configures and trains the stacking regressor pool
    ├── evaluation.py           # Computes MAE, RMSE, and R² validation scores
    ├── visualization.py        # Generates interactive Plotly browser charts (No popups)
    └── prediction.py           # Handles payload formatting for live UI predictions
