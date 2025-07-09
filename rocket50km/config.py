# config.py

# Physics constants
GRAVITY = 9.81        # m/s²
ISP = 280             # seconds (typical for small engines)
THRUST = 120_000      # N - tuned for ~3.5g liftoff with lighter rocket
RETRO_THRUST = -60_000  # N - for landing burn
DRAG_COEFF = 0.45
CROSS_SECTION_AREA = 1.0  # m² - small slender rocket

# Masses
PROPELLANT_MASS = 1000   # kg
DRY_MASS = 400           # kg

# Retro burn altitude
RETRO_BURN_ALT = 3000  # m

# Time step
TIME_STEP = 1  # s

# G-force limits
MAX_G_FORCE = 3.5  # g (limit for human-safe flight)