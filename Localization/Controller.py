import numpy as np
from PID_Controller import Integrator_Plant_PID_Controller

class Controller:
    def __init__(self, dt:float, v_min:float, v_max:float, w_max:float):
        # Control time constant (Two sample updates)
        time_constant = 2*dt
        # Control bandwidth (Nyquist frequency)
        bandwidth = 4 * np.pi/dt

        # Create PID Controllers
        self.linear = Integrator_Plant_PID_Controller(bandwidth, time_constant, dt)
        self.angular = Integrator_Plant_PID_Controller(bandwidth, time_constant, dt)

        self.dt = dt
        self.xd_prev = 0
        self.yd_prev = 0
        self.thetad_prev = 0

        self.v_min = v_min
        self.v_max = v_max
        self.w_max = w_max
        self.w_min = -w_max

    def compute_control_inputs_position(self, x:float, y:float, theta:float, xd:float, yd:float):
        # Compute feedforward control
        xd_dot = (xd - self.xd_prev) / self.dt
        yd_dot = (yd - self.yd_prev) / self.dt

        if abs(xd_dot) > 0 or abs(yd_dot > 0):
            thetad = np.atan2(yd_dot, xd_dot)
        else:
            thetad = self.thetad_prev

        thetad_dot = (thetad - self.thetad_prev) / self.dt
        
        self.xd_prev = xd
        self.yd_prev = yd
        self.thetad_prev = thetad

        # Feedforward control inputs
        vff = xd_dot * np.cos(theta) + yd_dot * np.sin(theta)
        wff = thetad_dot

        # Compute feedback control
        dx = xd - x
        dy = yd - y
        d = np.sqrt(dx*dx + dy*dy)

        # Not at target position, change desired theta to point at target
        thetad_target_pos = np.atan2(dy, dx)
        
        # Calculate Error in orientation to move toward target position
        etheta = thetad_target_pos - theta

        # Calculate Error in position (possible movement direction projected desired direction)
        ed = d * np.cos(etheta)
        
        # Make error in range [-pi, pi]
        while etheta > np.pi:
            etheta -= 2*np.pi
        while etheta < -np.pi:
            etheta += 2*np.pi

        # Calculate control inputs: Feedback Control + Feedforward Control
        v = self.linear.compute_input(ed) + vff
        w = self.angular.compute_input(etheta) + wff

        # Saturate
        if v > self.v_max:
            v = self.v_max
        if v < self.v_min:
            v = self.v_min
        if w > self.w_max:
            w = self.w_max
        if w < self.w_min:
            w = self.w_min

        return v,w
    
    def compute_control_inputs_angle(self, x:float, y:float, theta:float, xd:float, yd:float, thetad:float):
        # Compute feedforward control
        xd_dot = (xd - self.xd_prev) / self.dt
        yd_dot = (yd - self.yd_prev) / self.dt

        thetad_dot = (thetad - self.thetad_prev) / self.dt
        
        self.xd_prev = xd
        self.yd_prev = yd
        self.thetad_prev = thetad

        # Feedforward control inputs
        vff = xd_dot * np.cos(theta) + yd_dot * np.sin(theta)
        wff = thetad_dot

        # Compute feedback control
        dx = xd - x
        dy = yd - y
        d = np.sqrt(dx*dx + dy*dy)
        
        # Calculate Error in orientation to move toward target position
        etheta = thetad - theta

        # Calculate Error in position (possible movement direction projected desired direction)
        ed = d * np.cos(etheta)
        
        # Make error in range [-pi, pi]
        while etheta > np.pi:
            etheta -= 2*np.pi
        while etheta < -np.pi:
            etheta += 2*np.pi

        # Calculate control inputs: Feedback Control + Feedforward Control
        v = self.linear.compute_input(ed) + vff
        w = self.angular.compute_input(etheta) + wff

        # Saturate
        if v > self.v_max:
            v = self.v_max
        if v < self.v_min:
            v = self.v_min
        if w > self.w_max:
            w = self.w_max
        if w < self.w_min:
            w = self.w_min

        return v,w
