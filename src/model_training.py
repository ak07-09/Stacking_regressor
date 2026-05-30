import joblib
import pandas as pd
import os
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from src.data_preprocessing import preprocess_data

def train_stacking_model():
    print("⏳ Resolving absolute system directories...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
    
    target_file = None
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if 'student' in file.lower() and file.endswith('.csv'):
                target_file = os.path.join(DATA_DIR, file)
                break

    if not target_file:
        raise FileNotFoundError(f"❌ Could not find a student performance CSV inside: {DATA_DIR}")

    print(f"✅ Scanning complete! Loading: {target_file}")
    df = pd.read_csv(target_file)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Construct blending ensemble
    base_regressors = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=4))
    ]
    
    stacking_pipeline = StackingRegressor(
        estimators=base_regressors,
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking_pipeline)
    ])
    
    print("🚀 Running cross-validated training across Stacking estimators...")
    pipeline.fit(X_train, y_train)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"🎉 Stacking Regression Pipeline generated and saved: {MODEL_PATH}")

if __name__ == "__main__":
    train_stacking_model()