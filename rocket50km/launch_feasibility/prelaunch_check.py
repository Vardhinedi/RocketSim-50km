from launch_feasibility.model.predict import predict_launch_feasibility

def run_prelaunch_check():
    print("\n🧪 Running AI Prelaunch Feasibility Check...\n")

    # Example input values
    temp = 27
    wind = 12
    pressure = 1005
    humidity = 60
    visibility = 12
    clouds = 30
    thrust = 800
    pump_pressure = 100
    avionics = 0.98
    sensors = 0.96

    result = predict_launch_feasibility(
        temp=temp,
        wind=wind,
        pressure=pressure,
        humidity=humidity,
        visibility=visibility,
        clouds=clouds,
        thrust=thrust,
        pump_pressure=pump_pressure,
        avionics=avionics,
        sensors=sensors
    )

    # 🪛 Debug print
    print("Raw Result from AI:", result)

    # ✅ If model returned an error message instead of tuple
    if not isinstance(result, tuple):
        return {
            "error": result
        }

    try:
        return {
            "score": float(result[0].split(":")[1].split("±")[0].strip()),
            "uncertainty": float(result[0].split("±")[1].strip()),
            "decision": result[1].split(":")[1].strip(),
            "violations": result[2].split(":")[1].strip(),
            "alert": result[3],
            "plot_html": result[4]
        }
    except Exception as e:
        return {
            "error": f"Failed to parse result. Raw output: {result}. Error: {str(e)}"
        }
