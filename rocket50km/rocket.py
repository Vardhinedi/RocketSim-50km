# rocket.py

from config import *
import math

class Rocket:
    def __init__(self):
        self.time = 0
        self.altitude = 0
        self.velocity = 0
        self.mass = DRY_MASS + PROPELLANT_MASS
        self.burning = True
        self.meco = False
        self.parachute_deployed = False
        self.max_altitude = 0
        self.g_force = 0

    def get_air_density(self):
        return AIR_DENSITY_SEA_LEVEL * math.exp(-self.altitude / 8500)

    def get_drag_force(self):
        rho = self.get_air_density()
        drag = 0.5 * DRAG_COEFF * CROSS_SECTION_AREA * rho * self.velocity ** 2
        return -drag if self.velocity > 0 else drag

    def update(self):
        # Compute forces
        thrust = THRUST if self.burning else 0
        drag = self.get_drag_force()
        weight = self.mass * GRAVITY

        net_force = thrust + drag - weight
        acceleration = net_force / self.mass
        self.g_force = abs(acceleration / GRAVITY)

        # Update kinematics
        self.velocity += acceleration * TIME_STEP
        self.altitude += self.velocity * TIME_STEP
        self.altitude = max(0, self.altitude)
        self.time += TIME_STEP

        # Max altitude tracker
        self.max_altitude = max(self.max_altitude, self.altitude)

        # Fuel burn
        if self.burning:
            self.mass -= BURN_RATE * TIME_STEP
            if self.mass <= DRY_MASS:
                self.mass = DRY_MASS
                self.burning = False

        # MECO condition
        if not self.meco and (not self.burning or self.altitude >= MECO_ALTITUDE or self.g_force > MAX_G_FORCE):
            self.burning = False
            self.meco = True
            print(f"💥 MECO at t+{int(self.time)}s | Alt: {int(self.altitude)} m | V: {int(self.velocity)} m/s | G: {self.g_force:.2f} g")

        # Parachute deployment
        if not self.parachute_deployed and self.altitude < PARACHUTE_DEPLOY_ALT and self.velocity < 0:
            self.parachute_deployed = True
            print(f"🪂 Parachute deployed at t+{int(self.time)}s | Alt: {int(self.altitude)} m")

        # Apply parachute drag
        if self.parachute_deployed:
            self.velocity *= 0.8  # crude deceleration

        return {
            "t": self.time,
            "altitude": self.altitude,
            "velocity": self.velocity,
            "g_force": self.g_force,
            "meco": self.meco,
            "parachute": self.parachute_deployed
        }

    def has_landed(self):
        return self.altitude <= 0 and self.time > 3
