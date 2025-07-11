import math
import json
import os
import gym
from gym import spaces
import numpy as np

# Load configuration from JSON
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    cfg = json.load(f)

rocket_cfg = cfg["rocket"]
engine_cfg = cfg["engine"]
sim_cfg = cfg["simulation"]
env_cfg = cfg["environment"]

class RocketSimulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0.0
        self.altitude = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.throttle = 1.0
        self.fuel_mass = rocket_cfg["propellant_mass"]
        self.dry_mass = rocket_cfg["dry_mass"]
        self.mass = self.dry_mass + self.fuel_mass
        self.cross_section_area = rocket_cfg["cross_section_area"]
        self.drag_coeff = rocket_cfg["drag_coeff"]
        self.thrust = engine_cfg["thrust"]
        self.isp = engine_cfg["isp"]
        self.gravity = env_cfg["gravity"]
        self.air_density = env_cfg["air_density"]
        self.wind_speed = env_cfg["wind_speed"]
        self.min_throttle = engine_cfg["min_throttle"]
        self.time_step = sim_cfg["time_step"]
        self.max_sim_time = sim_cfg["max_sim_time"]

        self.done = False
        self.landed = False
        self.max_altitude = 0.0

        # Parachute settings
        self.parachute_deployed = False
        self.parachute_altitude = 10000  # deploy when falling below 10 km
        self.chute_drag_coeff = 20.0      # high drag with parachute

    def step(self, action):
        throttle = action[0] if isinstance(action, (list, np.ndarray)) else action
        self.update(throttle)
        return self.get_state()

    def get_state(self):
        return {
            "altitude": self.altitude,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "fuel_mass": self.fuel_mass,
        }

    def update(self, throttle=None):
        if self.done:
            return

        if throttle is not None:
            self.throttle = max(self.min_throttle, min(throttle, 1.0))

        # Check for parachute deployment
        if (not self.parachute_deployed 
            and self.altitude <= self.parachute_altitude 
            and self.velocity < 0):
            self.parachute_deployed = True
            self.drag_coeff = self.chute_drag_coeff  # big drag now!

        if self.fuel_mass > 0:
            flow_rate = self.thrust / (self.isp * self.gravity)
            fuel_burn = flow_rate * self.time_step * self.throttle
            if fuel_burn > self.fuel_mass:
                fuel_burn = self.fuel_mass
                self.throttle = 0.0
            self.fuel_mass -= fuel_burn
            self.mass = self.dry_mass + self.fuel_mass
            actual_thrust = self.thrust * self.throttle
        else:
            actual_thrust = 0.0
            self.throttle = 0.0
            self.mass = self.dry_mass

        # Drag force (direction opposes velocity)
        drag_force = 0.5 * self.air_density * self.velocity**2 * self.drag_coeff * self.cross_section_area
        drag_force *= -1 if self.velocity > 0 else 1

        net_force = actual_thrust - (self.mass * self.gravity) + drag_force
        self.acceleration = net_force / self.mass

        self.velocity += self.acceleration * self.time_step
        self.altitude += self.velocity * self.time_step
        self.time += self.time_step

        if self.altitude > self.max_altitude:
            self.max_altitude = self.altitude

        if self.altitude <= 0 and self.time > 2:
            self.altitude = 0
            self.velocity = 0
            self.acceleration = 0
            self.done = True
            self.landed = True


class RocketEnv(gym.Env):
    def __init__(self):
        super(RocketEnv, self).__init__()
        self.rocket = RocketSimulator()

        # Observation: [altitude, velocity, acceleration, fuel_mass]
        low = np.array([0.0, -500.0, -100.0, 0.0], dtype=np.float32)
        high = np.array([100000.0, 3000.0, 100.0, rocket_cfg["propellant_mass"]], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: throttle
        self.action_space = spaces.Box(
            low=np.array([engine_cfg["min_throttle"]], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.time_step = sim_cfg["time_step"]

    def reset(self, *, seed=None, options=None):
        self.rocket.reset()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.rocket.step(action)
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self.rocket.done
        truncated = self.rocket.time >= sim_cfg["max_sim_time"]
        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        status = "✅ Parachute" if self.rocket.parachute_deployed else "🟦 No Chute"
        print(f"Time: {self.rocket.time:.1f}s | Alt: {self.rocket.altitude:.1f} m | Vel: {self.rocket.velocity:.1f} m/s | Fuel: {self.rocket.fuel_mass:.1f} kg | {status}")

    def _get_obs(self):
        state = self.rocket.get_state()
        return np.array([
            state["altitude"],
            state["velocity"],
            state["acceleration"],
            state["fuel_mass"]
        ], dtype=np.float32)

    def _get_reward(self):
        if self.rocket.landed:
            return 200
        return (self.rocket.altitude / 1000.0) - (self.rocket.fuel_mass * 0.01)

    @property
    def spec(self):
        return None

    @property
    def unwrapped(self):
        return self
