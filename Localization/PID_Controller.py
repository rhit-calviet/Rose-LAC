import numpy as np

class Integrator_Plant_PID_Controller:
    def __init__(self, bandwidth:float, time_constant:float, dt:float):
        self.kd = 3 * bandwidth * time_constant - 1
        self.kp = 3 * bandwidth * bandwidth * time_constant
        self.ki = bandwidth * bandwidth * bandwidth * time_constant
        self.dt = dt

        self.alpha = dt / (dt + time_constant)

        self.e_prev = 0
        self.e_integral = 0

        self.u = 0

    def compute_input(self, e:float):
        # Derivative
        de = (e - self.e_prev) / self.dt
        self.e_prev = e

        # Integral
        self.e_integral += self.dt * e
        
        # Compute output
        u_des = self.kp * e + self.kd * de + self.ki * self.e_integral

        # Low pass filter
        self.u = self.alpha * u_des + (1 - self.alpha) * self.u

        return self.u
