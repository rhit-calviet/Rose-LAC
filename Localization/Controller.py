import numpy as np
from PID_Controller import Integrator_Plant_PID_Controller

class Controller:
    def __init__(self, dt:float, settling_time:float, v_min:float, v_max:float, w_max:float):
        time_constant = settling_time / 4
        self.linear = Integrator_Plant_PID_Controller(settling_time, time_constant, dt, v_min, v_max)
        self.angular = Integrator_Plant_PID_Controller(settling_time, time_constant, dt, -w_max, w_max)

    def compute_control_outputs(self, x, y, theta, xd, yd, thetad, epsilon_pos):
        dx = xd - x
        dy = yd - y
        d = np.sqrt(dx*dx + dy*dy)

        # Not at target position, change desired theta to point at target
        thetad_target_pos = np.atan2(dy, dx)
        
        # Calculate Error in orientation to move toward target position
        etheta = thetad_target_pos - theta

        # Calculate Error in position (possible movement direction projected desired direction)
        ed = d * np.cos(etheta)

        # If close enough to target position, turn to target orientation
        if d < epsilon_pos:
            etheta = thetad - theta
        
        # Make error in range [-pi, pi]
        while etheta > np.pi:
            etheta -= 2*np.pi
        while etheta < -np.pi:
            etheta += 2*np.pi

        # Calculate control inputs
        v = self.linear.output(ed)
        w = self.angular.output(etheta)

        return v, w
