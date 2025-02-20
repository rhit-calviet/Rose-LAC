import numpy as np

from Navigation.python.AccelerationLimitedProfile import AccelerationLimitedProile

'''
Come up with a desired x, y point for the rover to come to to begin the path to enter the charger. Pass those through to navigation which 
in turn return the actual x0, y0, and theta0 that it passes over control at. Use those values to navigate from, to the initial charger position.
'''

class EnterExitCharger:
    def __init__(self, x0:float, y0:float, theta0:float, x1:float, y1:float, theta1:float, runup_length:float, r:float, v_max:float, a_max:float, omega_max:float, alpha_max:float):
        self.accel_profile = AccelerationLimitedProile(v_max, a_max)
        self.omega_profile = AccelerationLimitedProile(omega_max, alpha_max)
        self.theta_prime = np.arctan2(self.y1 - self.y0, self.x1 - self.x0)
        self.t1 = self.omega_profile.total_time(self.theta_prime - self.theta0)
        self.t2 = self.accel_profile.total_time(np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2))
        self.t3 = self.omega_profile.total_time(self.theta1 - self.theta_prime)

    def step_enter_charger(self, t:float):
        '''
        phase:          turn, linear, radial turn, linear, no motion hold for setting, rotational hold
        control(flag): theta,    xy,       xy,       xy,               xy,                   theta
        '''
        if (t < self.t1):
            self.omega_profile.x(t, self.theta_prime - self.theta0)
            return (x, y, theta, flag)
        elif (t < self.t1 + self.t2):
            self.accel_profile.x(t - self.t1, np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2))
            return (x, y, theta, flag)
        elif (t < self.t1 + self.t2 + self.t3):
            self.omega_profile.x(t - self.t1 - self.t2, self.theta1 - self.theta_prime)
            return (x, y, theta, flag)
        else:
            return (x, y, theta, theta_flag)
    

    def step_enter_charger(self, t:float):
        if (t < self.t1):
            self.omega_profile.x(t, self.theta_prime - self.theta0)
            return (x, y, theta, flag)
        elif (t < self.t1 + self.t2):
            self.accel_profile.x(t - self.t1, np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2))
            return (x, y, theta, flag)
        elif (t < self.t1 + self.t2 + self.t3):
            self.omega_profile.x(t - self.t1 - self.t2, self.theta1 - self.theta_prime)
            return (x, y, theta, flag)
        else:
            return (x, y, theta, theta_flag)


    def desired_position(self, x0:float, y0:float, theta0:float, x1:float, y1:float, theta1:float, runup_length:float, r:float):
        '''
        Given the initial and final positions of the rover, the desired position is calculated.
        '''
        x = x1 - runup_length*np.cos(theta1)
        y = y1 - runup_length*np.sin(theta1)
        return (x, y, theta1)


    def path_start:
        '''
        Given the desired position, the path is generated.
        '''
        return 0