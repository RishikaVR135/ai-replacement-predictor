import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ml_engine import predict_student

def render_student_view():
    st.header("🎓 Student Placement Diagnostic Engine")
    st.write("Input your academic, technical, and soft skill profile to compute your placement readiness probability and receive an explainable diagnostic breakdown.")

    st.markdown("---")

    # 1. Profile Configuration Section
    st.subheader("1. Profile Parameters")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        backlogs = st.number_input("History of Backlogs", min_value=0, max_value=10, value=0)

    with col2:
        coding_score = st.slider("Coding Assessment Score (0-100)", min_value=0, max_value=100, value=65)
        aptitude_score = st.slider("Logical Aptitude Score (0-100)", min_value=0, max_value=100, value=70)

    with col3:
        soft_skills = st.slider("Soft Skills Rating (1-5)", min_value=1, max_value=5, value=3)
        certifications = st.number_input("Verified Certifications", min_value=0, max_value=10, value=1)

    # Compile input dictionary matching ml_engine structure
    student_data = {
        'cgpa': cgpa,
        'backlogs': backlogs,
        'coding_score': coding_score,
        'aptitude_score': aptitude_score,
        'soft_skills': soft_skills,
        'certifications': certifications
    }

    st.markdown("---")

    # 2. Prediction & Status Outputs
    prob, status, contribs = predict_student(student_data)

    st.subheader("2. Placement Probability & Status")
    
    metric_col1, metric_col2 = st.columns([1, 2])

    with metric_col1:
        st.metric(label="Placement Probability", value=f"{prob}%")
        
        # Color-coded status badge
        if status == "Ready":
            st.success(f"Status: **{status}**")
        elif status == "Near-Ready":
            st.warning(f"Status: **{status}**")
        else:
            st.error(f"Status: **{status}**")

    with metric_col2:
        # Gauge chart for visual score presentation
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Readiness Index"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 50], 'color': "#ff4b4b"},
                    {'range': [50, 75], 'color': "#ffa15e"},
                    {'range': [75, 100], 'color': "#2ca02c"}
                ],
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # 3. Explainable AI (SHAP Contributions)
    st.subheader("3. Explainable AI (XAI) Factor Transparency")
    st.write("Detailed breakdown showing parameter contributions toward your placement probability score.")

    df_contribs = pd.DataFrame(list(contribs.items()), columns=['Feature', 'Contribution'])
    df_contribs['Impact'] = df_contribs['Contribution'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    df_contribs = df_contribs.sort_values(by='Contribution', ascending=True)

    fig_shap = px.bar(
        df_contribs,
        x='Contribution',
        y='Feature',
        orientation='h',
        color='Impact',
        color_discrete_map={'Positive': '#2ca02c', 'Negative': '#ff4b4b'},
        title="SHAP Feature Contribution Scores"
    )
    fig_shap.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # 4. Automated Skill Gap Diagnostic & Roadmap
    st.subheader("4. Actionable Skill Gap Diagnostic & Upskilling Roadmap")
    
    st.markdown("#### Recommended Action Items:")
    
    roadmap_items = []
    if backlogs > 0:
        roadmap_items.append("⚠️ **Clear Active Backlogs:** Priority 1 for campus eligibility filters.")
    if coding_score < 70:
        roadmap_items.append("💻 **Enhance Coding Benchmark:** Complete Data Structures & Algorithms modules to push score above 70.")
    if certifications < 2:
        roadmap_items.append("📜 **Obtain Verified Certifications:** Complete at least 1 industry-recognized certification (e.g., AWS, Full-Stack, Data Analytics).")
    if soft_skills < 4:
        roadmap_items.append("🗣️ **Soft Skill Refinement:** Enroll in departmental communication & mock interview workshops.")
    
    if not roadmap_items:
        st.success("🎉 Excellent profile! You meet high-tier placement criteria. Focus on advanced mock interviews and system design practice.")
    else:
        for item in roadmap_items:
            st.write(item)

# Allow independent testing of this file
if __name__ == "__main__":
    render_student_view()
