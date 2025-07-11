import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from ai.rocket_env import RocketEnv
import gymnasium as gym
import gym as old_gym

# Custom wrapper to convert gym spaces to gymnasium spaces
class GymToGymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Convert action space
        if isinstance(self.action_space, old_gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=self.action_space.low,
                high=self.action_space.high,
                shape=self.action_space.shape,
                dtype=self.action_space.dtype
            )
        # Convert observation space
        if isinstance(self.observation_space, old_gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=self.observation_space.low,
                high=self.observation_space.high,
                shape=self.observation_space.shape,
                dtype=self.observation_space.dtype
            )

def train_ai():
    def make_env():
        def _init():
            env = RocketEnv()
            env = GymToGymnasiumWrapper(env)  # Apply wrapper
            return Monitor(env)
        return _init

    # Create environment
    env = DummyVecEnv([make_env()])

    # Evaluation environment
    eval_env = DummyVecEnv([make_env()])

    # Callbacks for training
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True
    )

    # PPO model configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],
            activation_fn=torch.nn.ReLU,
            ortho_init=True
        )
    )

    # Train the model
    model.learn(total_timesteps=500_000, callback=eval_callback, progress_bar=True)

    # Save model
    model.save("ai/rocket_policy_final")
    print("✅ Training complete and model saved!")

if __name__ == "__main__":
    os.makedirs("best_model", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    train_ai()