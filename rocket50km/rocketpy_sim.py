from rocketpy import Environment, SolidMotor, Rocket, Flight

# 🌍 Define the environment
Env = Environment(
    latitude=13.0827,
    longitude=80.2707,
    elevation=60,               # Elevation in meters
    date=(2025, 7, 14, 12)      # YYYY, MM, DD, HH (UTC)
)

Env.set_atmospheric_model(type="Forecast", file="GFS")
Env.info()

# 🔥 Define the solid motor
Pro75M1670 = SolidMotor(
    thrustSource=1300,              # Approximate constant thrust in N
    burnOut=3.9,                    # Burn time in seconds
    grainNumber=5,
    grainSeparation=5/1000,
    grainDensity=1815,
    grainOuterRadius=33/1000,
    grainInitialInnerRadius=15/1000,
    grainInitialHeight=120/1000,
    nozzleRadius=33/1000,
    throatRadius=11/1000,
    interpolationMethod="linear"
)

# 🚀 Define the rocket
RocketObj = Rocket(
    motor=Pro75M1670,
    radius=127/2000,
    mass=19.197 - 2.956,
    inertiaI=6.60,
    inertiaZ=0.0351,
    distanceRocketNozzle=-1.255,
    distanceRocketPropellant=-0.85704,
    powerOffDrag=5,
    powerOnDrag=2,
    environment=Env
)

# Add rocket components
RocketObj.add_nose(length=0.55829, kind="vonKarman", distance_to_cm=0.71971)
RocketObj.add_fins(4, span=0.100, root_chord=0.120, tip_chord=0.040, distance_to_cm=-1.04956)
RocketObj.add_tail(top_radius=0.0635, bottom_radius=0.0435, length=0.060, distance_to_cm=-1.194656)

# Set rail buttons (important for rail departure speed check)
RocketObj.set_rail_buttons([0.2, -0.5])

# 🧠 Flight simulation — move rail length here
FlightSim = Flight(
    rocket=RocketObj,
    environment=Env,
    inclination=90,
    heading=0,
    railLength=5.2  # Moved here!
)

# 🖨️ Print simulation results
print("\n🔎 Summary:")
print(f"🚀 Apogee: {FlightSim.apogee:.2f} m")
print(f"🕐 Time to apogee: {FlightSim.timeApogee:.2f} s")
print(f"📉 Maximum Speed: {FlightSim.maxSpeed:.2f} m/s")
print(f"📈 Maximum Acceleration: {FlightSim.maxAcceleration:.2f} m/s²")

# 📊 Optional: plot results
FlightSim.all_info()
