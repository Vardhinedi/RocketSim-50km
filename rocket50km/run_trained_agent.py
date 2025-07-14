# run_trained_agent.py

import time
import numpy as np
from stable_baselines3 import PPO
from ai.realistic_env import RealisticRocketEnv

# Load environment and model
env = RealisticRocketEnv()
model = PPO.load("ppo_rocket_final")

obs, _ = env.reset()
done = False

print("🚀 Launching rocket with trained PPO agent...\n")
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.05)  # simulate real-time
