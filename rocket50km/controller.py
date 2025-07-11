import numpy as np

class PIDController:
    def __init__(self, kp=0.5, ki=0.1, kd=0.2):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.prev_error = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return np.clip(output, 0, 1)  # Throttle range

class AIController:
    def __init__(self, model_path="ai/policy.h5"):
        from tensorflow import keras
        self.model = keras.models.load_model(model_path)

    def decide_throttle(self, state):
        # state = [altitude, velocity, acceleration]
        return self.model.predict(np.array([state]))[0][0]