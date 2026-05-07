import joblib

print("Prediction File")

model = joblib.load(
    "models/lightgbm_model.pkl"
)

print(model)