# rocket.py

from config import *
import math

class Rocket:
    def __init__(self):
        self.time = 0
        self.altitude = 0
        self.velocity = 0
        self.propellant_mass = PROPELLANT_MASS
        self.mass = DRY_MASS + self.propellant_mass
        self.max_altitude = 0
        self.acceleration = 0
        self.g_force = 0
        self.burnout = False
        self.retro_burn = False
        self.log = []

    def get_air_density(self):
        return 1.225 * math.exp(-self.altitude / 8500)

    def get_drag(self):
        rho = self.get_air_density()
        drag_mag = 0.5 * rho * DRAG_COEFF * CROSS_SECTION_AREA * self.velocity ** 2
        return -drag_mag if self.velocity > 0 else drag_mag

    def update(self):
        # Engine phase
        if self.propellant_mass > 0:
            thrust = THRUST
            mdot = THRUST / (ISP * GRAVITY)
            self.propellant_mass = max(0, self.propellant_mass - mdot * TIME_STEP)
        else:
            thrust = 0
            if not self.burnout:
                print("💥 MECO")
                self.burnout = True

        # Retro burn
        if self.burnout and self.altitude <= RETRO_BURN_ALT and not self.retro_burn and self.velocity < 0:
            print("🛬 Retro Burn Initiated")
            thrust = RETRO_THRUST
            self.retro_burn = True
        elif self.retro_burn:
            thrust = RETRO_THRUST

        # Update physics
        self.mass = DRY_MASS + self.propellant_mass
        drag = self.get_drag()
        self.acceleration = (thrust + drag) / self.mass - GRAVITY
        self.g_force = min(abs(self.acceleration / GRAVITY), MAX_G_FORCE)

        self.velocity += self.acceleration * TIME_STEP
        self.altitude += self.velocity * TIME_STEP
        self.altitude = max(self.altitude, 0)
        self.max_altitude = max(self.max_altitude, self.altitude)

        self.log.append((
            self.time, self.altitude, self.velocity, self.acceleration, self.g_force
        ))

        self.time += TIME_STEP
