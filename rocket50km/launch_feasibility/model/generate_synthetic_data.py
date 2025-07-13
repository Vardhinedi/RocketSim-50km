# launch_feasibility/model/generate_synthetic_data.py

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 1000  # Number of synthetic samples

data = {
    "Temperature_C": np.random.uniform(-10, 40, N),
    "Wind_Speed_kmph": np.random.uniform(0, 50, N),
    "Atmospheric_Pressure_hPa": np.random.uniform(950, 1050, N),
    "Humidity_percent": np.random.uniform(10, 100, N),
    "Visibility_km": np.random.uniform(1, 20, N),
    "Cloud_Cover_percent": np.random.uniform(0, 100, N),
    "Engine_Thrust_kN": np.random.uniform(500, 1000, N),
    "Fuel_Pump_Pressure_bar": np.random.uniform(50, 150, N),
    "Avionics_Status": np.random.uniform(0.9, 1.0, N),
    "Sensor_Reliability": np.random.uniform(0.85, 1.0, N),
}

# Assign outcomes based on simple heuristic
def label(row):
    if (
        0 <= row["Temperature_C"] <= 35 and
        row["Wind_Speed_kmph"] <= 30 and
        970 <= row["Atmospheric_Pressure_hPa"] <= 1030 and
        row["Humidity_percent"] <= 85 and
        row["Visibility_km"] >= 5 and
        row["Cloud_Cover_percent"] <= 70 and
        row["Engine_Thrust_kN"] >= 600 and
        row["Fuel_Pump_Pressure_bar"] >= 70 and
        row["Avionics_Status"] >= 0.95 and
        row["Sensor_Reliability"] >= 0.9
    ):
        return 1  # Successful
    else:
        return 0  # Failed

df = pd.DataFrame(data)
df["Outcome"] = df.apply(label, axis=1)

# Save as CSV
output_path = "launch_feasibility/model/synthetic_launch_data.csv"
df.to_csv(output_path, index=False)
print(f"✅ Saved {len(df)} synthetic samples to {output_path}")
