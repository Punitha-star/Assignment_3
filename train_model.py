import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

print("Training Models")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    "data/processed/MLcleanedFeature_full_data.csv"
)

# =========================
# X AND y
# =========================
y = df['status']

X = df.drop(
    'status',
    axis=1
)

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()

for col in X.select_dtypes(include='object').columns:

    X[col] = label_encoder.fit_transform(
        X[col].astype(str)
    )

print("Encoding Completed")

# =========================
# DUMMY ENCODING
# =========================
X = pd.get_dummies(
    X,
    drop_first=True
)

# =========================
# CREATE PROCESSED FOLDER
# =========================
os.makedirs(
    "data/processed",
    exist_ok=True
)

# =========================
# SAVE MODEL DATA
# =========================
model_df = pd.concat(
    [X, y],
    axis=1
)

model_df.to_csv(
    "data/processed/model_full_data.csv",
    index=False
)

print("Model Data Saved Successfully")

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =========================
# LOGISTIC REGRESSION
# =========================
lr = LogisticRegression(
    max_iter=1000
)

lr.fit(
    X_train_scaled,
    y_train
)

print("Logistic Regression Trained")

# =========================
# RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

print("Random Forest Trained")

# =========================
# LIGHTGBM
# =========================
lgb = LGBMClassifier()

lgb.fit(
    X_train,
    y_train
)

print("LightGBM Trained")

# =========================
# CREATE MODELS FOLDER
# =========================
os.makedirs(
    "models",
    exist_ok=True
)

# =========================
# SAVE MODELS
# =========================
joblib.dump(
    lr,
    "models/logistic_model.pkl"
)

joblib.dump(
    rf,
    "models/random_forest_model.pkl"
)

joblib.dump(
    lgb,
    "models/lightgbm_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# =========================
# SAVE ENCODER
# =========================
joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

print("Encoder Saved")

print("All Models Saved Successfully")