# rocketpy_env/generate_env_profile.py

from rocketpy import Environment
import json

# 🌍 Define the environment (use your location)
Env = Environment(latitude=13.0827, longitude=80.2707)  # Chennai
Env.set_date((2025, 7, 14, 12))  # YYYY, M, D, Hour (UTC)
Env.set_atmospheric_model(type="Forecast", file="GFS")

# 📊 Extract air density and wind speed profile up to 50km
air_density_profile = {}
wind_speed_profile = {}

for altitude in range(0, 51000, 1000):  # Every 1 km up to 50 km
    air_density = Env.density(altitude)  # already float
    wind_speed = Env.wind_speed(altitude)  # already float
    air_density_profile[str(altitude)] = round(air_density, 5)
    wind_speed_profile[str(altitude)] = round(wind_speed, 3)

# 🧪 Bundle into a dictionary
profile_data = {
    "air_density": air_density_profile,
    "wind_speed": wind_speed_profile
}

# 💾 Save to file
output_path = "rocketpy_env/env_profile.json"
with open(output_path, "w") as f:
    json.dump(profile_data, f, indent=2)

print("✅ Environment profile saved to:", output_path)
