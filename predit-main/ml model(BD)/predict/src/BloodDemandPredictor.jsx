import axios from "axios";
import React, { useState } from "react";

function BloodDemandPredictor() {
  const [formData, setFormData] = useState({
    district: "",
    bloodGroup: "",
    total_donations: "",
    available_units: "",
    is_festival_week: "",
    is_holiday: "",
    is_monsoon: "",
    population_density: "",
    hospital_count: "",
    accident_rate: "",
    avg_temperature: "",
    donor_registration_trend: ""
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", formData);
      setPrediction(res.data.predicted_demand);
    } catch (error) {
      console.error("Prediction error:", error);
      // More detailed logging to help diagnose network / server errors
      if (error.response) {
        // Server responded with a non-2xx status
        console.error("Response status:", error.response.status);
        console.error("Response data:", error.response.data);
      } else if (error.request) {
        // Request made but no response received
        console.error("No response received — request:", error.request);
      } else {
        // Something happened setting up the request
        console.error("Axios setup error:", error.message);
      }
      alert("Error fetching prediction — check console/network tab for details");
    }
  };

  return (
    <div className="container">
      <h2>🩸 Blood Demand Predictor</h2>
      <form onSubmit={handleSubmit}>
        {Object.keys(formData).map((key) => (
          <div className="form-row" key={key}>
            <label className="form-label" htmlFor={key}>{key.replace(/_/g, ' ')}:</label>
            <input
              id={key}
              className="form-input"
              name={key}
              value={formData[key]}
              onChange={handleChange}
              type="text"
              required
            />
          </div>
        ))}
        <button className="predict-button" type="submit">Predict</button>
      </form>
      {prediction && (
        <div className="prediction-result">
          <h3>Predicted Demand: {prediction.toFixed(2)} units</h3>
        </div>
      )}
    </div>
  );
}

export default BloodDemandPredictor;
