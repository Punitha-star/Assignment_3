import pandas as pd
from sqlalchemy import create_engine

print("Uploading To SQL")

# LOAD CLEANED FEATURE DATA
df = pd.read_csv(
    "data/processed/MLcleanedFeature_full_data.csv"
)

# MYSQL CONNECTION
engine = create_engine(
    "mysql+pymysql://root:12345@localhost/Project"
)

# UPLOAD TO MYSQL
df.to_sql(
    "employee_job_acceptance",
    con=engine,
    if_exists="replace",
    index=False
)

print("Data Uploaded Successfully")