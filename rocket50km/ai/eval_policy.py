# ai/eval_policy.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import traceback
from launch_feasibility.model.predict import predict_launch_feasibility
from stable_baselines3 import PPO
from ai.realistic_env import RealisticRocketEnv as RocketEnv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def evaluate_policy():
    print("\n🧪 Running AI Prelaunch Feasibility Check...")

    # Load launch input
    try:
        with open("launch_input.json", "r") as f:
            launch_data = json.load(f)
    except Exception as e:
        print("❌ Failed to load launch_input.json:", e)
        return

    # Run model prediction
    try:
        result = predict_launch_feasibility(**launch_data)
    except Exception as e:
        print("❌ Unexpected error during prediction:", e)
        traceback.print_exc()
        return

    if not isinstance(result, dict):
        print("\n❌ Unexpected error during evaluation:", result)
        return

    # Pretty output
    print("\n" + "━" * 63)
    print(f"🧠 AI Prelaunch Decision: {result['decision']}")
    print(f"📊 Score:       {result['score']}  (Threshold: {result['threshold']})")
    print(f"📉 Uncertainty: {result['uncertainty']}")
    print(f"⚠️  Violations: {len(result['violations'])}")
    print(f"📢 Alert:\n{result['alert']}")
    print("━" * 63)

    if "No go" in result["decision"]:
        print("\n❌ Launch aborted due to feasibility check failure.")
        return

    print("\n🚀 Launch Feasible — Initiating RL Simulation...")

    # RL SIMULATION PART
    env = RocketEnv()
    model = PPO.load("best_model/best_model")
    obs, _ = env.reset()
    done = False

    times, altitudes, velocities, fuels, x_positions = [], [], [], [], []

    plt.style.use("dark_background")
    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 4, figure=fig)

    alt_ax = fig.add_subplot(gs[0, :2])
    vel_ax = fig.add_subplot(gs[1, :2])
    fuel_ax = fig.add_subplot(gs[2, :2])
    traj_ax = fig.add_subplot(gs[:2, 2:])
    telemetry_ax = fig.add_subplot(gs[2, 2:])
    telemetry_ax.axis("off")

    text_box = telemetry_ax.text(0, 1.0, "", va='top', ha='left',
                                 fontsize=12, fontfamily='monospace', color='white')

    alt_line, = alt_ax.plot([], [], label="Altitude (m)", color="cyan")
    vel_line, = vel_ax.plot([], [], label="Velocity (m/s)", color="lime")
    fuel_line, = fuel_ax.plot([], [], label="Fuel Mass (kg)", color="red")
    traj_line, = traj_ax.plot([], [], label="Trajectory", color="orange")

    for ax in [alt_ax, vel_ax, fuel_ax, traj_ax]:
        ax.grid(True)
        ax.legend()

    alt_ax.set_ylabel("Altitude (m)")
    vel_ax.set_ylabel("Velocity (m/s)")
    fuel_ax.set_ylabel("Fuel (kg)")
    fuel_ax.set_xlabel("Time (s)")
    traj_ax.set_xlabel("Downrange Distance (m)")
    traj_ax.set_ylabel("Altitude (m)")
    traj_ax.set_title("🚀 Rocket Trajectory")

    update_interval = 3
    frame = 0

    while not done:
        loop_start = time.time()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        r = env.rocket
        t, alt, vel, acc, fuel, x = r.time, r.altitude, r.velocity, r.acceleration, r.fuel_mass, r.downrange
        chute = "DEPLOYED 🪂" if r.parachute_deployed else "Not Deployed"
        throttle = r.throttle

        times.append(t)
        altitudes.append(alt)
        velocities.append(vel)
        fuels.append(fuel)
        x_positions.append(x)

        if frame % update_interval == 0:
            alt_line.set_data(times, altitudes)
            vel_line.set_data(times, velocities)
            fuel_line.set_data(times, fuels)
            traj_line.set_data(x_positions, altitudes)

            for ax in [alt_ax, vel_ax, fuel_ax, traj_ax]:
                ax.relim()
                ax.autoscale_view()

            text_box.set_text(
                f"📡  LIVE TELEMETRY\n"
                f"---------------------------\n"
                f"⏱️   Time:         {t:6.1f} s\n"
                f"🛰️   Altitude:     {alt:6.1f} m\n"
                f"🚀  Velocity:     {vel:6.1f} m/s\n"
                f"📈  Acceleration: {acc:6.2f} m/s²\n"
                f"⛽  Fuel Left:    {fuel:6.1f} kg\n"
                f"🎚️   Throttle:     {throttle:.2f}\n"
                f"🪂  Parachute:    {chute}"
            )

            plt.pause(0.001)

        frame += 1
        time.sleep(max(0, env.time_step - (time.time() - loop_start)))
        done = terminated or truncated

    print(f"\n✅ Final Altitude: {env.rocket.max_altitude:.1f} m")
    print(f"✅ Flight Duration: {env.rocket.time:.1f} s")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    evaluate_policy()
