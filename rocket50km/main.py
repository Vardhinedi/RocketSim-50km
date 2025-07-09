import time
import math
from config import (GRAVITY, ISP, THRUST, DRAG_COEFF, CROSS_SECTION_AREA,
                    PROPELLANT_MASS, DRY_MASS, RETRO_THRUST, RETRO_BURN_ALT,
                    TIME_STEP, MAX_G_FORCE)

def get_air_density(altitude):
    return 1.225 * math.exp(-altitude / 8500)

def calculate_drag(velocity, altitude):
    rho = get_air_density(altitude)
    return 0.5 * rho * DRAG_COEFF * CROSS_SECTION_AREA * velocity**2 * (-1 if velocity > 0 else 1)

def main():
    time_elapsed = 0
    altitude = 0
    velocity = 0
    propellant_mass = PROPELLANT_MASS
    total_mass = DRY_MASS + propellant_mass
    max_altitude = 0

    burnout = False
    retro_burn_started = False
    g_force = 0  # Initialize for first print

    print("🚀 Launching Rocket...\n")

    while True:
        # Print state BEFORE physics update
        print(f"t+{time_elapsed:3}s | Alt: {altitude:7.1f} m | V: {velocity:7.1f} m/s | G: {g_force:4.2f} g")

        # --- PHYSICS UPDATE ---
        drag = calculate_drag(velocity, altitude)

        # Thrust or MECO
        if propellant_mass > 0:
            thrust_force = THRUST
            mdot = THRUST / (ISP * GRAVITY)
            propellant_mass -= mdot * TIME_STEP
            propellant_mass = max(propellant_mass, 0)
        else:
            if not burnout:
                print("\n💥 MECO - Main Engine Cutoff\n")
                burnout = True
            thrust_force = 0

        # Retro burn
        if burnout and altitude <= RETRO_BURN_ALT and not retro_burn_started and velocity < 0:
            print("🛬 Retro Burn Initiated\n")
            thrust_force = RETRO_THRUST
            retro_burn_started = True
        elif retro_burn_started:
            thrust_force = RETRO_THRUST

        # Update dynamics
        total_mass = DRY_MASS + propellant_mass
        acceleration = (thrust_force + drag) / total_mass - GRAVITY
        g_force = min(abs(acceleration / GRAVITY), MAX_G_FORCE)

        velocity += acceleration * TIME_STEP
        altitude += velocity * TIME_STEP
        altitude = max(0, altitude)

        max_altitude = max(max_altitude, altitude)

        # Landing condition
        if altitude <= 0 and time_elapsed > 3:
            break

        time.sleep(TIME_STEP)
        time_elapsed += TIME_STEP

    print("\n✅ Flight Complete")
    print(f"🛰 Max Altitude: {max_altitude / 1000:.2f} km")

if __name__ == "__main__":
    main()
