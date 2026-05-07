# =========================
# IMPORT LIBRARIES
# =========================
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# =========================
# CREATE OUTPUTS FOLDER
# =========================
os.makedirs(
    "outputs",
    exist_ok=True
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Job Acceptance Prediction System",
    layout="wide"
)

# =========================
# TITLE
# =========================
st.title("💼 Job Acceptance Prediction System")

st.markdown(
    "👨‍💼 Candidate Performance | 📈 Placement Insights | 🤖 ML Analytics"
)

# =========================
# DATABASE CONNECTION
# =========================
engine = create_engine(
    "mysql+pymysql://root:12345@localhost/Project"
)

# =========================
# LOAD DATA
# =========================
df = pd.read_sql(
    "SELECT * FROM employee_job_acceptance",
    con=engine
)

# =========================
# CLEAN COLUMN NAMES
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Navigation")

module = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard KPI",
        "EDA & ML Analytics",
        "Dataset Preview"
    ]
)

# =========================
# DASHBOARD KPI
# =========================
if module == "Dashboard KPI":

    st.subheader("📊 Key KPIs in Streamlit Dashboards")

    # =========================
    # KPI CALCULATIONS
    # =========================

    # Total Candidates
    total_candidates = len(df)

    # Placement Rate (%)
    placement_rate = round(
        (df['status'] == 1).mean() * 100,
        2
    )

    # Job Acceptance Rate (%)
    job_acceptance_rate = round(
            (df['status'] == 1).mean() * 100,
            2
        ) 
        

    # Average Interview Score
    avg_interview_score = round(
        df['interview_score'].mean(),
        2
    )

    # Average Skills Match %
    avg_skills = round(
        df['skills_match_percentage'].mean(),
        2
    )

    # Offer Dropout Rate
    dropout_rate = round(
        (df['status'] == 0).mean() * 100,
        2
    )

    # High-Risk Candidate Percentage
    high_risk_percentage = round(
        (
            (
                (df['interview_score'] < 50) &
                (df['skills_match_percentage'] < 50)
            ).sum()
            / len(df)
        ) * 100,
        2
    )

    # =========================
    # KPI DISPLAY
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Candidates",
        total_candidates
    )

    col2.metric(
        "Placement Rate (%)",
        placement_rate
    )

    col3.metric(
        "Job Acceptance Rate (%)",
        job_acceptance_rate
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Average Interview Score",
        avg_interview_score
    )

    col5.metric(
        "Average Skills Match %",
        avg_skills
    )

    col6.metric(
        "Offer Dropout Rate (%)",
        dropout_rate
    )

    st.metric(
        "High-Risk Candidate Percentage",
        high_risk_percentage
    )

    # =========================
    # PLACEMENT DISTRIBUTION
    # =========================

    st.subheader("📈 Placement Distribution")

    fig, ax = plt.subplots(figsize=(7,5))

    sns.countplot(
        x='status',
        data=df,
        ax=ax
    )

    plt.title("Placement Distribution")

    # SAVE IMAGE
    plt.savefig(
        "outputs/placement_distribution.png"
    )

    st.pyplot(fig)

# =========================
# DATASET PREVIEW
# =========================
elif module == "Dataset Preview":

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(20))

# =========================
# EDA & ML ANALYTICS
# =========================
elif module == "EDA & ML Analytics":

    st.subheader("📊 Analyst Tasks (EDA & ML Analytics)")

    analysis = st.selectbox(
        "Select Analysis",
        [
            "Interview Score vs Job Acceptance",
            "Skills Match Percentage Impact",
            "Company Tier vs Acceptance Rate",
            "Experience vs Placement Probability",
            "Competition Level vs Job Acceptance",
            "Academic Scores vs Placement Outcome",
            "Skills Match vs Interview Performance",
            "Certification Impact on Job Acceptance",
            "Employability Test Score Analysis",
            "Correlation Heatmap"
        ]
    )

    # =========================
    # Interview Score vs Job Acceptance
    # =========================
    if analysis == "Interview Score vs Job Acceptance":

        st.markdown("""
        ### 📌 Analysis
        
        EDA is performed to uncover patterns
        between interview score and job acceptance.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.boxplot(
            x='status',
            y='interview_score',
            data=df,
            ax=ax
        )

        plt.title("Interview Score vs Job Acceptance")

        plt.savefig(
            "outputs/interview_score_vs_acceptance.png"
        )

        st.pyplot(fig)

    # =========================
    # Skills Match Percentage Impact
    # =========================
    elif analysis == "Skills Match Percentage Impact":

        st.markdown("""
        ### 📌 Analysis
        
        Skills match percentage impact on placement.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.scatterplot(
            x='skills_match_percentage',
            y='placement_score',
            hue='status',
            data=df,
            ax=ax
        )

        plt.title("Skills Match Percentage Impact")

        plt.savefig(
            "outputs/skills_match_impact.png"
        )

        st.pyplot(fig)

    # =========================
    # Company Tier vs Acceptance Rate
    # =========================
    elif analysis == "Company Tier vs Acceptance Rate":

        st.markdown("""
        ### 📌 Analysis
        
        Company tier versus acceptance rate analysis.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.countplot(
            x='company_tier',
            hue='status',
            data=df,
            ax=ax
        )

        plt.title("Company Tier vs Acceptance Rate")

        plt.xticks(rotation=20)

        plt.savefig(
            "outputs/company_tier_acceptance.png"
        )

        st.pyplot(fig)

    # =========================
    # Experience vs Placement Probability
    # =========================
    elif analysis == "Experience vs Placement Probability":

        st.markdown("""
        ### 📌 Analysis
        
        Experience versus placement probability.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.boxplot(
            x='status',
            y='years_of_experience',
            data=df,
            ax=ax
        )

        plt.title("Experience vs Placement Probability")

        plt.savefig(
            "outputs/experience_vs_placement.png"
        )

        st.pyplot(fig)

    # =========================
    # Competition Level vs Job Acceptance
    # =========================
    elif analysis == "Competition Level vs Job Acceptance":

        st.markdown("""
        ### 📌 Analysis
        
        Competition level versus job acceptance.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.countplot(
            x='competition_level',
            hue='status',
            data=df,
            ax=ax
        )

        plt.title("Competition Level vs Job Acceptance")

        plt.savefig(
            "outputs/competition_level_acceptance.png"
        )

        st.pyplot(fig)

    # =========================
    # Academic Scores vs Placement Outcome
    # =========================
    elif analysis == "Academic Scores vs Placement Outcome":

        st.markdown("""
        ### 📌 Analysis
        
        Academic scores versus placement outcome.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.boxplot(
            x='status',
            y='academic_score',
            data=df,
            ax=ax
        )

        plt.title("Academic Scores vs Placement Outcome")

        plt.savefig(
            "outputs/academic_score_placement.png"
        )

        st.pyplot(fig)

    # =========================
    # Skills Match vs Interview Performance
    # =========================
    elif analysis == "Skills Match vs Interview Performance":

        st.markdown("""
        ### 📌 Analysis
        
        Skills match versus interview performance.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.scatterplot(
            x='skills_match_percentage',
            y='interview_score',
            hue='status',
            data=df,
            ax=ax
        )

        plt.title("Skills Match vs Interview Performance")

        plt.savefig(
            "outputs/skills_vs_interview.png"
        )

        st.pyplot(fig)

    # =========================
    # Certification Impact on Job Acceptance
    # =========================
    elif analysis == "Certification Impact on Job Acceptance":

        st.markdown("""
        ### 📌 Analysis
        
        Certification impact on job acceptance.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.boxplot(
            x='status',
            y='certifications_count',
            data=df,
            ax=ax
        )

        plt.title("Certification Impact on Job Acceptance")

        plt.savefig(
            "outputs/certification_impact.png"
        )

        st.pyplot(fig)

    # =========================
    # Employability Test Score Analysis
    # =========================
    elif analysis == "Employability Test Score Analysis":

        st.markdown("""
        ### 📌 Analysis
        
        Employability and placement score distribution.
        """)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.histplot(
            df['placement_score'],
            kde=True,
            ax=ax
        )

        plt.title("Employability Test Score Analysis")

        plt.savefig(
            "outputs/employability_test.png"
        )

        st.pyplot(fig)

    # =========================
    # Correlation Heatmap
    # =========================
    elif analysis == "Correlation Heatmap":

        st.markdown("""
        ### 📌 Analysis
        
        Correlation analysis among numeric features.
        """)

        numeric_df = df.select_dtypes(include='number')

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12,8))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=False,
            ax=ax
        )

        plt.title("Correlation Heatmap")

        plt.savefig(
            "outputs/correlation_heatmap.png"
        )

        st.pyplot(fig)