from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model + scaler
model = pickle.load(open("Model.pkl", "rb"))
scaler = pickle.load(open("standar_scaler.pkl", "rb"))

# ⚠️ MUST MATCH YOUR TRAINING ORDER (31 FEATURES)
FEATURE_ORDER = [
    "tenure_trim",
    "MonthlyCharges_trim",
    "TotalCharges_mode_trim",

    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",

    "MultipleLines_No_phone_service",
    "MultipleLines_Yes",

    "InternetService_Fiber_optic",
    "InternetService_No",

    "OnlineSecurity_No_internet_service",
    "OnlineSecurity_Yes",

    "OnlineBackup_No_internet_service",
    "OnlineBackup_Yes",

    "DeviceProtection_No_internet_service",
    "DeviceProtection_Yes",

    "TechSupport_No_internet_service",
    "TechSupport_Yes",

    "StreamingTV_No_internet_service",
    "StreamingTV_Yes",

    "StreamingMovies_No_internet_service",
    "StreamingMovies_Yes",

    "PaperlessBilling_Yes",

    "Payment_Credit_card",
    "Payment_Electronic_check",
    "Payment_Mailed_check",

    "telecom_BSNL",
    "telecom_Jio",
    "telecom_VI",

    "Contract_One_year"
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            features = []

            for col in FEATURE_ORDER:
                val = request.form.get(col, 0)
                features.append(float(val))

            X = np.array([features])

            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]

            prediction = (
                "Customer Will Churn ❌"
                if pred == 0 else
                "Customer Will Stay ✅"
            )

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)