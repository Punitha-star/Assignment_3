import os
import time

print("=" * 60)
print("🚀 JOB ACCEPTANCE PREDICTION PROJECT")
print("=" * 60)

# =========================
# STEP 1 - DATA CLEANING
# =========================
print("\n📌 STEP 1: Running Data Cleaning & Feature Engineering...\n")

cleaning_status = os.system(
    "python scripts/data_cleaning.py"
)

if cleaning_status == 0:
    print("✅ Data Cleaning Completed Successfully\n")
else:
    print("❌ Error in Data Cleaning\n")

time.sleep(2)

# =========================
# STEP 2 - MODEL TRAINING
# =========================
print("\n📌 STEP 2: Training Machine Learning Models...\n")

training_status = os.system(
    "python scripts/train_model.py"
)

if training_status == 0:
    print("✅ Model Training Completed Successfully\n")
else:
    print("❌ Error in Model Training\n")

time.sleep(2)

# =========================
# STEP 3 - SQL UPLOAD
# =========================
print("\n📌 STEP 3: Uploading Data To MySQL...\n")

sql_status = os.system(
    "python scripts/database_upload.py"
)

if sql_status == 0:
    print("✅ Data Uploaded To MySQL Successfully\n")
else:
    print("❌ Error in SQL Upload\n")

time.sleep(2)

# =========================
# PROJECT COMPLETED
# =========================
print("=" * 60)
print("🎉 PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\n📊 You can now run Streamlit Dashboard using:\n")

print(
    "streamlit run app/MLstreamlit_app.py"
)