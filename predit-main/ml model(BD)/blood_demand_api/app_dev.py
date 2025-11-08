from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Mock Blood Demand Prediction API Running ✅"})

@app.route('/predict', methods=['POST'])
def predict():
    # Lightweight mock prediction endpoint for frontend development when model dependencies
    # (xgboost, joblib, etc.) can't be installed. Returns a deterministic, simple prediction
    # based on numeric fields in the request so the frontend can be tested.
    try:
        data = request.get_json(force=True) or {}
        # A simple heuristic: sum numeric-looking fields, fallback to 10.0
        total = 0.0
        for v in data.values():
            try:
                total += float(v)
            except Exception:
                continue
        predicted = total if total > 0 else 10.0
        return jsonify({"predicted_demand": float(predicted), "status": "mocked"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
