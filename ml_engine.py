import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# Generate synthetic dataset matching PDF dimensions
def generate_synthetic_data(n_samples=100):
    np.random.seed(42)
    df = pd.DataFrame({
        'student_id': [f"STU_{1000+i}" for i in range(n_samples)],
        'branch': np.random.choice(['CSE', 'ISE', 'ECE', 'MECH'], size=n_samples),
        'cgpa': np.round(np.random.uniform(5.5, 9.8, size=n_samples), 2),
        'backlogs': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'coding_score': np.random.randint(40, 100, size=n_samples),
        'aptitude_score': np.random.randint(40, 100, size=n_samples),
        'soft_skills': np.random.randint(1, 6, size=n_samples),
        'certifications': np.random.randint(0, 5, size=n_samples)
    })
    
    # Generate synthetic target label based on scores
    score = (df['cgpa'] * 10) + df['coding_score'] * 0.4 + df['aptitude_score'] * 0.3 - (df['backlogs'] * 15)
    df['placed'] = (score > 100).astype(int)
    return df

# Train XGBoost Model
df_batch = generate_synthetic_data()
X = df_batch.drop(columns=['student_id', 'branch', 'placed'])
y = df_batch['placed']

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)
explainer = shap.TreeExplainer(model)

def predict_student(input_dict):
    df_input = pd.DataFrame([input_dict])
    prob = float(model.predict_proba(df_input)[0][1] * 100)
    
    if prob >= 75:
        status = "Ready"
    elif prob >= 50:
        status = "Near-Ready"
    else:
        status = "Needs Training"
        
    shap_vals = explainer.shap_values(df_input)
    contribs = dict(zip(df_input.columns, shap_vals[0]))
    
    return round(prob, 2), status, contribs
