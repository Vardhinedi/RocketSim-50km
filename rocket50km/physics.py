import math
import numpy as np

class Atmosphere:
    @staticmethod
    def get_density(altitude):
        """US Standard Atmosphere 1976 model with exponential decay"""
        if altitude < 11000:  # Troposphere
            return 1.225 * (1 - 0.0065 * altitude / 288.15)**4.256
        elif altitude < 25000:  # Lower stratosphere
            return 0.3639 * math.exp(-(altitude - 11000) / 6341.7)
        else:  # Upper atmosphere
            return 0.08803 * (25000 / altitude)**1.22

class DragCalculator:
    @staticmethod
    def calculate(velocity, altitude, area, cd, is_parachute=False):
        rho = Atmosphere.get_density(altitude)
        drag_area = area * (50 if is_parachute else 1)
        drag_force = 0.5 * rho * cd * drag_area * velocity * abs(velocity)
        return -drag_force if velocity > 0 else drag_force