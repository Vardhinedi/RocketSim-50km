# train_rl_agent.py

import os
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from ai.realistic_env import RealisticRocketEnv  # Use your actual env import

# Create the environment
env = RealisticRocketEnv()

# Set up PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01
)

# Save checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="ppo_rocket"
)

# Train the agent
model.learn(
    total_timesteps=200_000,  # You can increase this
    callback=checkpoint_callback
)

# Save the final model
model.save("ppo_rocket_final")

print("✅ Training complete. Model saved as 'ppo_rocket_final.zip'")
