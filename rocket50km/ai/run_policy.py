import os
import numpy as np
from rocket import Rocket
from ai.rocket_env import RocketEnv

def run_policy():
    # Load the trained policy
    model_path = os.path.join(os.path.dirname(__file__), "rocket_policy_final.zip")
    
    # Create environment
    env = RocketEnv()
    rocket = env.rocket  # Get the rocket instance
    
    # Run the policy
    obs, _ = env.reset()
    while True:
        action = np.array([[0.8]])  # Example fixed action, replace with your policy
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # Print state information
        print(f"Altitude: {rocket.altitude:.1f}m | "
              f"Velocity: {rocket.velocity:.1f}m/s | "
              f"Throttle: {rocket.throttle:.2f} | "
              f"Mass: {rocket.mass:.1f}kg")
        
        if terminated or truncated:
            break

if __name__ == "__main__":
    run_policy()