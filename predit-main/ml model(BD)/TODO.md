# TODO: Correct Features and Processing for Blood Demand Predictor

## Tasks
- [x] Update blood_demand_api/app.py: Load scaler, label encoders, feature names. In /predict route, apply label encoding to categoricals, scale numericals, ensure feature order matches model.
- [x] Update predict/src/BloodDemandPredictor.jsx: Replace generic fields with specific ones for each feature, using appropriate input types.
- [ ] Test the prediction by running the backend and frontend, submitting a form, and verifying output.
