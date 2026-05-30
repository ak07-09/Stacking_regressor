import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # 🧼 Drop columns that aren't useful features or will break regression
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
    if 'grade' in df.columns:
        df = df.drop(columns=['grade'])
        
    # 🎯 TARGET SELECTION: Explicitly map to total_score
    if 'total_score' in df.columns:
        X = df.drop(columns=['total_score'])
        y = df['total_score'].astype(float)
    else:
        raise KeyError("Could not find the continuous target variable 'total_score' in your CSV dataset.")
    
    # Identify feature types dynamically
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor