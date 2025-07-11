import time
from ai.rocket_env import RocketSimulator

rocket = RocketSimulator()
print("🚀 Launching...\n")

while not rocket.landed and rocket.time < rocket.max_sim_time:
    # Simple logic: full throttle until 20s, then cut off
    throttle = 1.0 if rocket.time < 20 else 0.0

    telemetry = rocket.step([throttle])

    # Print every 2 seconds
    if int(rocket.time * 10) % 20 == 0:
        print(
            f"Time: {telemetry['time']:.1f}s | Altitude: {telemetry['altitude']:.2f} m | "
            f"Velocity: {telemetry['velocity']:.2f} m/s | Fuel: {telemetry['fuel_mass']:.1f} kg"
        )

    time.sleep(rocket.time_step)

print("\n🛬 Landed.")
print(f"✅ Max altitude reached: {rocket.max_altitude:.2f} meters")
