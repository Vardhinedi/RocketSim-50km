# launch_feasibility/model/predict.py

import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import joblib
from tensorflow.keras.models import load_model

# Load model and artifacts
model = load_model("launch_feasibility/model/launch_model.keras")
scaler = joblib.load("launch_feasibility/model/scaler.save")
y_train = np.load("launch_feasibility/model/y_train.npy")

# Thresholds for validation
ranges = {
    "Temperature_C": (-10, 40),
    "Wind_Speed_kmph": (0, 50),
    "Atmospheric_Pressure_hPa": (950, 1050),
    "Humidity_percent": (10, 100),
    "Visibility_km": (1, 20),
    "Cloud_Cover_percent": (0, 100),
    "Engine_Thrust_kN": (500, 1000),
    "Fuel_Pump_Pressure_bar": (50, 150),
    "Avionics_Status": (0.9, 1.0),
    "Sensor_Reliability": (0.85, 1.0)
}

thresholds = {
    "Temperature_C": {"min": 0, "max": 35},
    "Wind_Speed_kmph": {"max": 30},
    "Atmospheric_Pressure_hPa": {"min": 970, "max": 1030},
    "Humidity_percent": {"max": 85},
    "Visibility_km": {"min": 5},
    "Cloud_Cover_percent": {"max": 70},
    "Engine_Thrust_kN": {"min": 600},
    "Fuel_Pump_Pressure_bar": {"min": 70},
    "Avionics_Status": {"min": 0.95},
    "Sensor_Reliability": {"min": 0.90}
}

is_manned = True
# === FIXED: Static 0.5 threshold ===
dynamic_threshold = 0.5

def predict_launch_feasibility(temp, wind, pressure, humidity, visibility, clouds, thrust, pump_pressure, avionics, sensors):
    timesteps = 10
    features = 11
    time = np.linspace(0, 1, timesteps)

    # Initial input dictionary
    inputs = {
        "Temperature_C": temp,
        "Wind_Speed_kmph": wind,
        "Atmospheric_Pressure_hPa": pressure,
        "Humidity_percent": humidity,
        "Visibility_km": visibility,
        "Cloud_Cover_percent": clouds,
        "Engine_Thrust_kN": thrust,
        "Fuel_Pump_Pressure_bar": pump_pressure,
        "Avionics_Status": avionics,
        "Sensor_Reliability": sensors
    }

    # Validate inputs
    for feature, value in inputs.items():
        min_val, max_val = ranges.get(feature, (-float('inf'), float('inf')))
        if not (min_val <= value <= max_val):
            return f"Error: {feature} value {value} out of range [{min_val}, {max_val}]"

    # Simulate 10 time steps
    input_data = np.zeros((timesteps, features))
    for i, t in enumerate(time):
        input_data[i] = [
            temp + 5 * np.sin(2 * np.pi * t),
            wind + 3 * np.sin(2 * np.pi * t + 0.5),
            pressure + 10 * np.sin(2 * np.pi * t + 1),
            humidity + 5 * np.sin(2 * np.pi * t + 1.5),
            visibility + 2 * np.sin(2 * np.pi * t + 2),
            clouds + 5 * np.sin(2 * np.pi * t + 2.5),
            thrust + 10 * np.sin(2 * np.pi * t + 3),
            pump_pressure + 5 * np.sin(2 * np.pi * t + 3.5),
            avionics + 0.01 * np.sin(2 * np.pi * t + 4),
            sensors + 0.01 * np.sin(2 * np.pi * t + 4.5),
            (temp + 5 * np.sin(2 * np.pi * t)) * (wind + 3 * np.sin(2 * np.pi * t + 0.5))
        ]

    # Scale input
    input_scaled = scaler.transform(input_data)
    pred_input = input_scaled.reshape(1, timesteps, features)

    # Monte Carlo prediction
    predictions = [model.predict(pred_input, verbose=0)[0][0] for _ in range(10)]
    score = np.mean(predictions)
    uncertainty = np.std(predictions)

    # Check rule-based violations
    max_inputs = {k: np.max(input_data[:, i]) for i, k in enumerate(inputs)}
    min_inputs = {k: np.min(input_data[:, i]) for i, k in enumerate(inputs)}
    violations = []

    for feature in inputs:
        thresh = thresholds.get(feature, {})
        if "min" in thresh and min_inputs[feature] < thresh["min"]:
            violations.append(f"{feature}: {min_inputs[feature]:.2f} < {thresh['min']}")
        if "max" in thresh and max_inputs[feature] > thresh["max"]:
            violations.append(f"{feature}: {max_inputs[feature]:.2f} > {thresh['max']}")

    # Decision logic
    decision = "✅ Good to go" if score >= 0.25 else "❌ No go"

    alert = "🚨 RED ALERT 🚨\n" + "\n".join(violations) if violations else "✅ All systems within limits"

    # Optional: create base64-encoded plot
    plt.figure(figsize=(8, 4))
    plt.plot(time, input_data[:, 0], label="Temperature")
    plt.plot(time, input_data[:, 1], label="Wind Speed")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Environmental Trends")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return {
        "score": round(score, 4),
        "uncertainty": round(uncertainty, 4),
        "decision": decision,
        "threshold": round(dynamic_threshold, 4),
        "violations": violations,
        "alert": alert,
        "img_base64": img_str
    }
