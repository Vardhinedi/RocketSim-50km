import math
import gym
from gym import spaces
import numpy as np
import json
import os
import csv

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    cfg = json.load(f)

rocket_cfg = cfg["rocket"]
engine_cfg = cfg["engine"]
sim_cfg = cfg["simulation"]
env_cfg = cfg["environment"]

# Load RocketPy-generated atmosphere
env_profile_path = os.path.join(os.path.dirname(__file__), '..', 'rocketpy_env', 'env_profile.json')
with open(env_profile_path, 'r') as f:
    env_profile = json.load(f)

density_lookup = {int(k): v for k, v in env_profile["air_density"].items()}
wind_lookup = {int(k): v for k, v in env_profile["wind_speed"].items()}

LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'flight_log.csv')

class RealisticRocketSim:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0.0
        self.altitude = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.throttle = 1.0
        self.pitch_angle = 90.0  # auto-updated during gravity turn
        self.downrange = 0.0
        self.horizontal_velocity = 0.0

        self.fuel_mass = rocket_cfg["propellant_mass"]
        self.dry_mass = rocket_cfg["dry_mass"]
        self.mass = self.fuel_mass + self.dry_mass

        self.cross_section_area = rocket_cfg["cross_section_area"]
        self.drag_coeff = rocket_cfg["drag_coeff"]
        self.base_drag_coeff = rocket_cfg["drag_coeff"]
        self.chute_drag_coeff = 15.0

        self.thrust = engine_cfg["thrust"]
        self.isp = engine_cfg["isp"]
        self.gravity = env_cfg["gravity"]
        self.min_throttle = engine_cfg["min_throttle"]

        self.time_step = sim_cfg["time_step"]
        self.max_sim_time = sim_cfg["max_sim_time"]

        self.parachute_deployed = False
        self.parachute_altitude = 10000

        self.done = False
        self.landed = False
        self.max_altitude = 0.0

        # Reset CSV log
        with open(LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "altitude", "velocity", "acceleration", "fuel_mass",
                "pitch_angle", "throttle", "parachute", "downrange", "horizontal_velocity"
            ])

    def get_air_density(self, altitude):
        alt = int(min(50000, max(0, round(altitude, -3))))
        return density_lookup.get(alt, 1.2)

    def get_wind_speed(self, altitude):
        alt = int(min(50000, max(0, round(altitude, -3))))
        return wind_lookup.get(alt, 0.0)

    def get_gravity(self, altitude):
        R = 6371000
        return 9.80665 * (R / (R + altitude)) ** 2

    def update_pitch(self):
        if self.altitude < 1000:
            self.pitch_angle = 90.0
        elif self.altitude < 10000:
            self.pitch_angle = 90.0 - 45.0 * ((self.altitude - 1000) / 9000.0)
        else:
            self.pitch_angle = 45.0

    def step(self, action):
        if self.done:
            return self.get_state()

        throttle = float(np.clip(action[0], self.min_throttle, 1.0))
        self.throttle = throttle

        self.update_pitch()

        if not self.parachute_deployed and self.altitude <= self.parachute_altitude and self.velocity < 0:
            self.parachute_deployed = True
            self.drag_coeff = self.chute_drag_coeff

        g = self.get_gravity(self.altitude)
        rho = self.get_air_density(self.altitude)
        wind_speed = self.get_wind_speed(self.altitude)

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

        pitch_rad = math.radians(self.pitch_angle)
        thrust_vertical = thrust_force * math.sin(pitch_rad)
        thrust_horizontal = thrust_force * math.cos(pitch_rad)

        # Vertical
        drag = 0.5 * rho * self.velocity**2 * self.drag_coeff * self.cross_section_area
        drag *= -1 if self.velocity > 0 else 1
        net_vertical_force = thrust_vertical - self.mass * g + drag
        self.acceleration = net_vertical_force / self.mass
        self.velocity += self.acceleration * self.time_step
        self.altitude += self.velocity * self.time_step

        # Horizontal
        relative_horizontal_velocity = self.horizontal_velocity - wind_speed
        drag_horizontal = 0.5 * rho * relative_horizontal_velocity**2 * self.drag_coeff * self.cross_section_area
        drag_horizontal *= -1 if relative_horizontal_velocity > 0 else 1
        net_horizontal_force = thrust_horizontal + drag_horizontal
        horiz_acc = net_horizontal_force / self.mass
        self.horizontal_velocity += horiz_acc * self.time_step
        self.downrange += self.horizontal_velocity * self.time_step

        self.time += self.time_step
        self.max_altitude = max(self.max_altitude, self.altitude)

        with open(LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round(self.time, 2),
                round(self.altitude, 2),
                round(self.velocity, 2),
                round(self.acceleration, 3),
                round(self.fuel_mass, 2),
                round(self.pitch_angle, 2),
                round(self.throttle, 2),
                int(self.parachute_deployed),
                round(self.downrange, 2),
                round(self.horizontal_velocity, 2)
            ])

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
            "pitch": self.pitch_angle
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
        print(f"Time: {self.rocket.time:.1f}s | Alt: {self.rocket.altitude:.1f} m | "
              f"Vel: {self.rocket.velocity:.1f} m/s | Fuel: {self.rocket.fuel_mass:.1f} kg | "
              f"Pitch: {self.rocket.pitch_angle:.1f}° | {chute}")

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
