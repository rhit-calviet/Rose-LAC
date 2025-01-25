import numpy as np

class Integrator_Plant_PID_Controller:
    def __init__(self, settling_time:float, time_constant:float, dt:float, u_min:float, u_max:float):
        pole = 4 / settling_time
        self.kd = 3 * pole * time_constant - 1
        self.kp = 3 * pole * pole * time_constant
        self.ki = pole * pole * pole * time_constant
        self.dt = dt

        self.alpha = dt / (dt + time_constant)

        self.e_prev = 0
        self.e_integral = 0
        self.integral_max = u_max / self.ki
        self.integral_min = u_min / self.ki

        self.u = 0
        self.u_min = u_min
        self.u_max = u_max

    def output(self, e:float):
        # Derivative
        de = (e - self.e_prev) / self.dt
        self.e_prev = e

        # Integral
        self.e_integral += self.dt * e
        if self.e_integral > self.integral_max:
            self.e_integral = self.integral_max
        if self.e_integral < self.integral_min:
            self.e_integral = self.integral_min
        
        # Compute output
        u_des = self.kp * e + self.kd * de + self.ki * self.e_integral

        # Low pass filter
        self.u = self.alpha * u_des + (1 - self.alpha) * self.u

        if self.u > self.u_max:
            self.u = self.u_max
        if self.u < self.u_min:
            self.u = self.u_min

        return self.u
