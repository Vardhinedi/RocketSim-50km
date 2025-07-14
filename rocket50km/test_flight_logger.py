from ai.realistic_env import RealisticRocketEnv

env = RealisticRocketEnv()
obs, _ = env.reset()

for _ in range(300):
    obs, reward, done, _, _ = env.step([0.8, 85.0])  # ✅ throttle + pitch
    env.render()
    if done:
        break
