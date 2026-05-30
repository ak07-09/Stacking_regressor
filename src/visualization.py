import plotly.express as px
import pandas as pd

def plot_hours_studied(df):
    return px.histogram(df, x='weekly_self_study_hours', nbins=15, title='Weekly Self-Study Hours Distribution', color_discrete_sequence=['#636EFA'])

def plot_attendance(df):
    return px.box(df, y='attendance_percentage', title='Class Attendance Percentages', color_discrete_sequence=['#EF553B'])

def plot_target_distribution(df):
    return px.histogram(df, x='total_score', title='Total Score Range (Regression Target)', color_discrete_sequence=['#00CC96'])

def plot_participation_distribution(df):
    if 'class_participation' in df.columns:
        return px.histogram(df, x='class_participation', title='Class Participation Scores', color_discrete_sequence=['#AB63FA'])
    return px.pie(df, names=df.select_dtypes(include='object').columns[0], title='Categorical Distribution')

def plot_heatmap(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    return px.imshow(corr, text_auto=".2f", title="Feature Correlation Heatmap", color_continuous_scale='RdBu_r')

def plot_predictions_vs_actual(y_test, y_pred):
    plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    fig = px.scatter(
        plot_df, x='Actual', y='Predicted', opacity=0.5,
        title='Predicted vs. Actual Academic Scores',
        labels={'Actual': 'Observed Total Score', 'Predicted': 'Estimated Total Score'}
    )
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="Red", dash="dash"))
    return fig

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plot_df = pd.DataFrame({'Predicted': y_pred, 'Residuals': residuals})
    fig = px.scatter(
        plot_df, x='Predicted', y='Residuals', opacity=0.5,
        title='Model Prediction Residual Deviations',
        labels={'Predicted': 'Fitted Values', 'Residuals': 'Residual Error'}
    )
    fig.add_shape(type="line", x0=y_pred.min(), y0=0, x1=y_pred.max(), y1=0, line=dict(color="black", dash="dash"))
    return fig