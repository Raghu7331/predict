# app.py — Blood Demand Prediction API (Final Version)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow React frontend (CORS)

# -----------------------------------------------------------
# 1️⃣ Load Model
# -----------------------------------------------------------
try:
    model = joblib.load("blood_demand_xgb_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# -----------------------------------------------------------
# 2️⃣ Load Optional Preprocessing Artifacts
# -----------------------------------------------------------
try:
    # These files are optional; handle gracefully if missing
    fitted_label_encoders = joblib.load("fitted_label_encoders.pkl")
    model_feature_names = joblib.load("model_feature_names.pkl")
    print("✅ Preprocessing artifacts loaded successfully!")
except Exception as e:
    print("⚠️ Skipping preprocessing artifacts:", e)
    fitted_label_encoders = {}
    model_feature_names = None

# -----------------------------------------------------------
# 3️⃣ Manual categorical mappings (if encoders not found)
# -----------------------------------------------------------
district_map = {
    "anantapur": 0, "chittoor": 1, "east godavari": 2, "guntur": 3,
    "krishna": 4, "kurnool": 5, "nellore": 6, "prakasam": 7,
    "srikakulam": 8, "visakhapatnam": 9, "vizianagaram": 10,
    "west godavari": 11, "ysr kadapa": 12
}

blood_group_map = {
    "o+": 0, "a+": 1, "b+": 2, "ab+": 3,
    "o-": 4, "a-": 5, "b-": 6, "ab-": 7
}

# -----------------------------------------------------------
# 4️⃣ Routes
# -----------------------------------------------------------

@app.route("/")
def home():
    return jsonify({"message": "Blood Demand Prediction API Running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Parse incoming JSON
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # ---------------------------------------------------
        # Clean categorical text → numeric codes
        # ---------------------------------------------------
        if "district" in df.columns:
            df["district"] = (
                df["district"].astype(str).str.strip().str.lower().map(district_map).fillna(-1)
            )

        if "bloodGroup" in df.columns:
            df["bloodGroup"] = (
                df["bloodGroup"].astype(str).str.strip().str.lower().map(blood_group_map).fillna(-1)
            )

        # ---------------------------------------------------
        # Reorder columns to match training feature names
        # ---------------------------------------------------
        if model_feature_names:
            # Fill any missing columns with 0
            for col in model_feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[model_feature_names]
        else:
            print("⚠️ Using dynamic columns:", list(df.columns))

        # ---------------------------------------------------
        # Make prediction
        # ---------------------------------------------------
        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_demand": float(prediction),
            "status": "success"
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
# 5️⃣ Run Flask App
# -----------------------------------------------------------
if __name__ == "__main__":
    import socket

    port = 5000
    while True:
        try:
            print(f"🚀 Starting Flask server on port {port}...")
            app.run(host="0.0.0.0", port=port, debug=True)
            break
        except OSError:
            print(f"⚠️ Port {port} in use, trying next...")
            port += 1
