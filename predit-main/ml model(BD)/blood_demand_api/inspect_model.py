import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder # Import for type hints and understanding

# --- Define file paths ---
MODEL_PATH = 'blood_demand_xgb_model.pkl'
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODERS_PATH = 'fitted_label_encoders.pkl'
FEATURE_NAMES_PATH = 'model_feature_names.pkl'

# --- 1. Load the XGBoost Model ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"\nSuccessfully loaded model from {MODEL_PATH}")
    print("Model type:", type(model))
    # You can inspect some model attributes
    print("Model booster:", model.booster)
    print("Model feature names during training (if set):", model.feature_names_in_)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# --- 2. Load the StandardScaler ---
try:
    scaler = joblib.load(SCALER_PATH)
    print(f"\nSuccessfully loaded StandardScaler from {SCALER_PATH}")
    print("Scaler type:", type(scaler))
    print("Number of features scaler was fitted on:", scaler.n_features_in_)
    print("Mean of features (first 5):", scaler.mean_[:5])
    print("Standard deviation of features (first 5):", scaler.scale_[:5])
    print("Feature names the scaler was fitted on:", scaler.feature_names_in_) # CRITICAL for checking consistency
except FileNotFoundError:
    print(f"Error: Scaler file not found at {SCALER_PATH}")
except Exception as e:
    print(f"Error loading scaler: {e}")

# --- 3. Load the Fitted Label Encoders (dictionary) ---
try:
    fitted_label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    print(f"\nSuccessfully loaded Label Encoders from {LABEL_ENCODERS_PATH}")
    print("Type of loaded object:", type(fitted_label_encoders))
    print("Number of loaded encoders:", len(fitted_label_encoders))
    for col, le in fitted_label_encoders.items():
        print(f"  Encoder for '{col}': type={type(le)}, classes={le.classes_}")
except FileNotFoundError:
    print(f"Error: Label Encoders file not found at {LABEL_ENCODERS_PATH}")
except Exception as e:
    print(f"Error loading label encoders: {e}")

# --- 4. Load the Model Feature Names List ---
try:
    model_feature_names = joblib.load(FEATURE_NAMES_PATH)
    print(f"\nSuccessfully loaded Model Feature Names from {FEATURE_NAMES_PATH}")
    print("Type of loaded object:", type(model_feature_names))
    print("Number of features:", len(model_feature_names))
    print("First 5 feature names:", model_feature_names[:5])
    print("Last 5 feature names:", model_feature_names[-5:])
except FileNotFoundError:
    print(f"Error: Model Feature Names file not found at {FEATURE_NAMES_PATH}")
except Exception as e:
    print(f"Error loading model feature names: {e}")

# --- Example of creating a dummy input for testing preprocessing ---
print("\n--- Testing Preprocessing Prototype with Dummy Data ---")

# Replace these with actual values that match your expected input structure
# and feature types (e.g., 'A+' for blood_group, actual numbers for age etc.)
dummy_input_data = {
    'age': 30,
    'blood_group': 'A+', # Example categorical, needs to be one of the trained classes
    'region': 'North',    # Example categorical
    'medical_emergency_flag': 0,
    'day_of_week_encoded': 1, # If you encoded day_of_week directly in training data
    'month_encoded': 1,
    'is_weekend': 0,
    'demand_quantity_lag_1': 150.5, # Example value
    'demand_quantity_lag_7': 140.2,
    'demand_quantity_roll_mean_7': 145.8,
    # Add ALL features your model expects based on MODEL_FEATURE_NAMES
}

# --- IMPORTANT: Reconstruct the ACTUAL_CATEGORICAL_COLS from your training script ---
# These should match the keys in your fitted_label_encoders dictionary
ACTUAL_CATEGORICAL_COLS = ['blood_group', 'region'] # <<< Adjust this to your actual categorical columns

try:
    input_df = pd.DataFrame([dummy_input_data])
    print("\nOriginal dummy input DataFrame:")
    print(input_df)

    # 1. Apply Label Encoding
    for col in ACTUAL_CATEGORICAL_COLS:
        if col in input_df.columns and col in fitted_label_encoders:
            print(f"Encoding '{col}': {input_df[col].iloc[0]} -> {fitted_label_encoders[col].transform(input_df[col]).iloc[0]}")
            input_df[col] = fitted_label_encoders[col].transform(input_df[col])
        elif col in input_df.columns:
            print(f"Warning: No LabelEncoder found for categorical column: {col}")


    # 2. Scale numerical features
    features_to_scale_for_prediction = [col for col in input_df.columns if col in scaler.feature_names_in_]
    if features_to_scale_for_prediction:
        print(f"\nScaling features: {features_to_scale_for_prediction}")
        # Make a copy to avoid SettingWithCopyWarning if input_df is a slice
        input_df_scaled_portion = input_df[features_to_scale_for_prediction].copy()
        input_df[features_to_scale_for_prediction] = scaler.transform(input_df_scaled_portion)
    else:
        print("\nNo features to scale identified by the loaded scaler.")

    print("\nDataFrame after Label Encoding and Scaling:")
    print(input_df)

    # 3. Ensure columns match the training data (order and presence)
    final_input_df = pd.DataFrame(columns=model_feature_names)
    for col in model_feature_names:
        if col in input_df.columns:
            final_input_df[col] = input_df[col]
        else:
            final_input_df[col] = 0 # Or a default value if more appropriate
            print(f"Warning: Feature '{col}' not found in dummy input, filled with 0.")

    print("\nFinal DataFrame ready for model.predict():")
    print(final_input_df)
    print("Column order:", final_input_df.columns.tolist())

    # Make a dummy prediction
    dummy_prediction = model.predict(final_input_df[model_feature_names])[0]
    print(f"\nDummy Prediction: {dummy_prediction:.4f}")

except Exception as e:
    print(f"\nError during dummy input preprocessing/prediction: {e}")