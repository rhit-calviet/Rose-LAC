import numpy as np
from PID_Controller import Integrator_Plant_PID_Controller

class Controller:
    def __init__(self, dt:float, v_min:float, v_max:float, w_max:float, zeta_v:float, wn_v:float, zeta_w:float, wn_w:float):

        # Create PID Controllers
        self.linear = Integrator_Plant_PID_Controller(dt, zeta_v, wn_v, v_min, v_max)
        self.angular = Integrator_Plant_PID_Controller(dt, zeta_w, wn_w, -w_max, w_max)

        self.dt = dt
        self.x_des_prev = 0
        self.y_des_prev = 0
        self.theta_des_prev = 0

        # Low pass filter
        self.xdot_des = 0
        self.ydot_des = 0
        self.thetadot_des = 0

        time_constant = 2*dt
        self.alpha = dt / (dt + time_constant)

    def compute_control_inputs(self, x:float, y:float, theta:float, xdot:float, ydot:float, thetadot:float, x_des:float, y_des:float, theta_des:float, angle_control:bool):
        # Compute derivative of desired coordinates
        xdot_des = (x_des - self.x_des_prev) / self.dt
        ydot_des = (y_des - self.y_des_prev) / self.dt
        # Low pass filter
        self.xdot_des = self.alpha * self.xdot_des + (1-self.alpha)*xdot_des
        self.ydot_des = self.alpha * self.ydot_des + (1-self.alpha)*ydot_des
        xdot_des = self.xdot_des
        ydot_des = self.ydot_des

        # Compute derivative of desired angle
        if not angle_control:
            if abs(xdot_des) > 0 or abs(ydot_des > 0):
                theta_des = np.atan2(ydot_des, xdot_des)
            else:
                theta_des = self.theta_des_prev

        thetadot_des = (theta_des - self.theta_des_prev) / self.dt
        # Low pass filter
        self.thetadot_des = self.alpha * self.theta_des_prev + (1 - self.alpha) * thetadot_des
        thetadot_des = self.thetadot_des
        
        self.x_des_prev = x_des
        self.y_des_prev = y_des
        self.theta_des_prev = theta_des

        # Desired Control Derivatives
        d_des_dot = xdot_des * np.cos(theta) + ydot_des * np.sin(theta)
        theta_des_dot = thetadot_des

        # Compute Desired Control Values
        dx = x_des - x
        dy = y_des - y

        # Not at target position, change desired theta to point at target
        if not angle_control:
            theta_des = np.atan2(dy, dx)
        
        # Calculate Error in orientation to move toward target position
        etheta = theta_des - theta
        # Make error in range [-pi, pi]
        while etheta > np.pi:
            etheta -= 2*np.pi
        while etheta < -np.pi:
            etheta += 2*np.pi
        
        etheta_dot = -thetadot

        # Calculate Error in position (possible movement direction projected desired direction)
        ed =      dx   * np.cos(theta) + dy   * np.sin(theta)
        ed_dot = -xdot * np.cos(theta) - ydot * np.sin(theta)

        # Calculate control inputs
        v = self.linear.compute_input(ed, ed_dot, d_des_dot)
        w = self.angular.compute_input(etheta, etheta_dot, theta_des_dot)

        return v,w
