import pandas as pd
import os

print("Running Data Cleaning")

# LOAD DATASET
df = pd.read_csv(
    "data/raw/HR_Job_Placement_Dataset.csv"
)

# CLEAN COLUMN NAMES
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

# REMOVE DUPLICATES
df = df.drop_duplicates()

# FILL NULL VALUES
for col in df.select_dtypes(include='number'):
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object'):
    df[col] = df[col].fillna(df[col].mode()[0])

# =========================
# FEATURE ENGINEERING
# =========================

# Academic Score
df['academic_score'] = (
    df['ssc_percentage'] +
    df['hsc_percentage'] +
    df['degree_percentage']
) / 3

# Interview Score
df['interview_score'] = (
    df['technical_score'] +
    df['aptitude_score'] +
    df['communication_score']
) / 3

# Placement Score
df['placement_score'] = (
    0.4 * df['interview_score'] +
    0.3 * df['skills_match_percentage'] +
    0.3 * df['academic_score']
)

# CLEAN STATUS
df['status'] = (
    df['status']
    .astype(str)
    .str.strip()
    .str.lower()
)

df['status'] = df['status'].map({
    'placed': 1,
    'not placed': 0,
    'yes': 1,
    'no': 0
})

# CREATE PROCESSED FOLDER
os.makedirs(
    "data/processed",
    exist_ok=True
)

# SAVE CLEANED DATA
df.to_csv(
    "data/processed/MLcleanedFeature_full_data.csv",
    index=False
)

print("Feature Engineering Completed")
print("MLcleanedFeature_full_data.csv")