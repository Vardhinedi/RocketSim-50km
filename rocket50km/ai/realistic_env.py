# realistic_env.py

import math
import gym
from gym import spaces
import numpy as np
import json
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    cfg = json.load(f)

rocket_cfg = cfg["rocket"]
engine_cfg = cfg["engine"]
sim_cfg = cfg["simulation"]
env_cfg = cfg["environment"]

class RealisticRocketSim:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0.0
        self.altitude = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.throttle = 1.0
        self.downrange = 0.0
        self.horizontal_velocity = 0.0

        self.fuel_mass = rocket_cfg["propellant_mass"]
        self.dry_mass = rocket_cfg["dry_mass"]
        self.mass = self.fuel_mass + self.dry_mass

        self.cross_section_area = rocket_cfg["cross_section_area"]
        self.drag_coeff = rocket_cfg["drag_coeff"]
        self.base_drag_coeff = rocket_cfg["drag_coeff"]
        self.chute_drag_coeff = 3.0

        self.thrust = engine_cfg["thrust"]
        self.isp = engine_cfg["isp"]
        self.gravity = env_cfg["gravity"]
        self.min_throttle = engine_cfg["min_throttle"]

        self.time_step = sim_cfg["time_step"]
        self.max_sim_time = sim_cfg["max_sim_time"]

        self.parachute_deployed = False
        self.parachute_altitude = 10000
        self.wind_speed = env_cfg["wind_speed"]

        self.done = False
        self.landed = False
        self.max_altitude = 0.0

        self.pitch_angle = 88.0  # degrees: slightly tilted for horizontal velocity

    def get_air_density(self, altitude):
        if altitude < 11000:
            temp = 288.15 - 0.0065 * altitude
            pressure = 101325 * (temp / 288.15) ** 5.2561
        else:
            temp = 216.65
            pressure = 22632 * math.exp(-0.0001577 * (altitude - 11000))
        return pressure / (287.05 * temp)

    def get_gravity(self, altitude):
        R = 6371000
        return 9.80665 * (R / (R + altitude)) ** 2

    def step(self, action):
        if self.done:
            return self.get_state()

        throttle = max(self.min_throttle, min(action[0], 1.0))

        # Deploy parachute
        if not self.parachute_deployed and self.altitude <= self.parachute_altitude and self.velocity < 0:
            self.parachute_deployed = True
            self.drag_coeff = self.chute_drag_coeff

        g = self.get_gravity(self.altitude)
        rho = self.get_air_density(self.altitude)

        if self.fuel_mass > 0:
            flow_rate = self.thrust / (self.isp * g)
            fuel_burn = flow_rate * self.time_step * throttle
            if fuel_burn > self.fuel_mass:
                fuel_burn = self.fuel_mass
                throttle = 0.0
            self.fuel_mass -= fuel_burn
            self.mass = self.dry_mass + self.fuel_mass
            thrust_force = self.thrust * throttle
        else:
            thrust_force = 0.0
            self.mass = self.dry_mass

        # Split thrust into vertical and horizontal
        pitch_rad = math.radians(self.pitch_angle)
        thrust_vertical = thrust_force * math.sin(pitch_rad)
        thrust_horizontal = thrust_force * math.cos(pitch_rad)

        drag = 0.5 * rho * self.velocity**2 * self.drag_coeff * self.cross_section_area
        drag *= -1 if self.velocity > 0 else 1

        net_vertical_force = thrust_vertical - self.mass * g + drag
        self.acceleration = net_vertical_force / self.mass
        self.velocity += self.acceleration * self.time_step
        self.altitude += self.velocity * self.time_step

        # Horizontal motion
        drag_horizontal = 0.5 * rho * self.horizontal_velocity**2 * self.drag_coeff * self.cross_section_area
        drag_horizontal *= -1 if self.horizontal_velocity > 0 else 1
        net_horizontal_force = thrust_horizontal + drag_horizontal
        horiz_acc = net_horizontal_force / self.mass
        self.horizontal_velocity += horiz_acc * self.time_step
        self.downrange += self.horizontal_velocity * self.time_step

        self.time += self.time_step
        self.max_altitude = max(self.max_altitude, self.altitude)

        if self.altitude <= 0 and self.time > 2:
            self.altitude = 0
            self.velocity = 0
            self.done = True
            self.landed = True

        return self.get_state()

    def get_state(self):
        return {
            "altitude": self.altitude,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "fuel_mass": self.fuel_mass,
            "x": self.downrange,
            "vx": self.horizontal_velocity,
        }

class RealisticRocketEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.rocket = RealisticRocketSim()
        self.observation_space = spaces.Box(
            low=np.array([0.0, -500.0, -100.0, 0.0], dtype=np.float32),
            high=np.array([100000.0, 3000.0, 100.0, rocket_cfg["propellant_mass"]], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([engine_cfg["min_throttle"]], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.time_step = sim_cfg["time_step"]

    def reset(self, *, seed=None, options=None):
        self.rocket.reset()
        return self._get_obs(), {}

    def step(self, action):
        self.rocket.step(action)
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self.rocket.done
        truncated = self.rocket.time >= self.rocket.max_sim_time
        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        chute = "✅ Parachute" if self.rocket.parachute_deployed else "🟦 No Chute"
        print(f"Time: {self.rocket.time:.1f}s | Alt: {self.rocket.altitude:.1f} m | Vel: {self.rocket.velocity:.1f} m/s | Fuel: {self.rocket.fuel_mass:.1f} kg | {chute}")

    def _get_obs(self):
        s = self.rocket.get_state()
        return np.array([s["altitude"], s["velocity"], s["acceleration"], s["fuel_mass"]], dtype=np.float32)

    def _get_reward(self):
        if self.rocket.landed:
            return 200
        return (self.rocket.altitude / 1000.0) - (self.rocket.fuel_mass * 0.01)

    @property
    def spec(self): return None
    @property
    def unwrapped(self): return self
