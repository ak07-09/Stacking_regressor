import pandas as pd

def make_prediction(model, weekly_self_study_hours, attendance_percentage, class_participation):
    # Constructing dataframe payload matching features expected by transformer pipeline
    input_data = pd.DataFrame([{
        'weekly_self_study_hours': weekly_self_study_hours,
        'attendance_percentage': attendance_percentage,
        'class_participation': class_participation
    }])
    
    predicted_score = model.predict(input_data)[0]
    
    # Floor and cap boundaries securely between standard 0 to 100 test scores
    return max(0.0, min(100.0, float(predicted_score)))