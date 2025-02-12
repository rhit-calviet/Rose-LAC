import numpy as np

class Integrator_Plant_PID_Controller:
    def __init__(self, dt:float, zeta:float, wn:float, u_min:float, u_max:float):
        w_samp = 2*np.pi/dt
        wn_max = w_samp / 5
        wn = min(wn_max,wn)
        self.kd = 25 / zeta
        self.kp = (1+self.kd)*(2*zeta*wn)
        self.ki = (1+self.kd)*wn*wn
        self.dt = dt

        self.alpha = dt / (dt + 1/wn)

        self.d_error = 0
        self.integral_error = 0
        self.integral_max = u_max / self.ki
        self.integral_min = u_min / self.ki


        self.u_min = u_min
        self.u_max = u_max

    def compute_input(self, error:float, error_dot:float, reference_dot:float):
        # Derivative Error
        self.d_error = self.alpha * self.d_error + (1-self.alpha)*error_dot
        # Integral Error
        self.integral_error += self.dt * error
        if self.integral_error > self.integral_max:
            self.integral_error = self.integral_max
        if self.integral_error < self.integral_min:
            self.integral_error = self.integral_min
        
        # Compute PID output
        u_pid = self.kp * error + self.kd * self.d_error + self.ki * self.integral_error

        # Compute Feedforward output
        u_ff = reference_dot

        # Compute total output
        u = u_pid + u_ff

        # Saturate output
        if u > self.u_max:
            u = self.u_max
        if u < self.u_min:
            u = self.u_min

        return u
