import json
import os
import numpy as np

class Rocket:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.reset()
    
    def reset(self):
        self.altitude = 0
        self.velocity = 0
        self.acceleration = 0
        self.propellant_mass = self.config["rocket"]["propellant_mass"]
        self.dry_mass = self.config["rocket"]["dry_mass"]
        self.throttle = 0
        self.time = 0
        self.earth_radius = 6.371e6
        self.g0 = 9.81
    
    @property
    def mass(self):
        return self.dry_mass + max(0, self.propellant_mass)
    
    def update(self, dt):
        config = self.config

        if self.propellant_mass > 0:
            thrust = self.throttle * config["engine"]["thrust"]
            isp = config["engine"]["isp"]
            fuel_rate = thrust / (isp * self.g0)
            fuel_consumed = min(fuel_rate * dt, self.propellant_mass)
            self.propellant_mass -= fuel_consumed
        else:
            thrust = 0

        gravity_force = self.g0 * (self.earth_radius / (self.earth_radius + self.altitude))**2 * self.mass

        scale_height = 8500
        air_density = 1.225 * np.exp(-self.altitude / scale_height)

        drag_coeff = config["rocket"]["drag_coeff"]
        cross_section = config["rocket"]["cross_section_area"]
        if self.altitude < config["stages"]["parachute_deploy_alt"] and self.velocity < 0:
            cross_section = 10.0  # parachute deployed

        drag = 0.5 * air_density * drag_coeff * cross_section * self.velocity**2
        drag_direction = -np.sign(self.velocity) if self.velocity != 0 else 0

        self.acceleration = (thrust - gravity_force + drag_direction * drag) / self.mass

        self.velocity += self.acceleration * dt
        self.altitude += self.velocity * dt

        if self.altitude <= 0:
            self.altitude = 0
            if self.velocity < 0:
                self.velocity = 0
                self.acceleration = 0

        self.time += dt

        return {
            "altitude": self.altitude,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "mass": self.mass,
            "time": self.time
        }

    def get_observation(self):
        target_altitude = self.config["simulation"]["target_altitude"]
        return np.array([
            min(self.altitude / target_altitude, 1.0),
            np.clip(self.velocity / 2000, -1.0, 1.0),
            np.clip(self.acceleration / (3 * self.g0), -1.0, 1.0)
        ], dtype=np.float32)
