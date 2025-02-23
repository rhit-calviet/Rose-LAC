import numpy as np
import math

from Navigation.python.AccelerationLimitedProfile import AccelerationLimitedProile
from Localization import InitialPosition

'''
Come up with a desired x, y point for the rover to come to to begin the path to enter the charger. Pass those through to navigation which 
in turn return the actual x0, y0, and theta0 that it passes over control at. Use those values to navigate from, to the initial charger position.
'''
# x0, y0, theta0: initial position and orientation of the rover
# x1, y1, theta1: final position and orientation of the rover
# runup_length: distance the rover should travel before entering the charger
# r: turn radius in rover path
# v_max: maximum velocity of the rover
# a_max: maximum acceleration of the rover
# omega_max: maximum angular velocity of the rover
# alpha_max: maximum angular acceleration of the rover
# flag: boolean value to indicate whether the rover is entering or exiting the charger

class EnterExitCharger:
    def __init__(self, x0:float, y0:float, theta0:float, x1:float, y1:float, theta1:float, runup_length:float, r:float, v_max:float, a_max:float, omega_max:float, alpha_max:float, flag:bool):
        self.accel_profile = AccelerationLimitedProile(v_max, a_max)
        self.omega_profile = AccelerationLimitedProile(omega_max, alpha_max)
        
        self.x0 = x0
        self.theta0 = theta0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.runup_length = runup_length
        self.theta1 = theta1
        self.p3y = y1 - runup_length * np.sin(theta1)
        self.p3x = x1 - runup_length * np.cos(theta1)
        self.theta_prime = np.arctan2(0, y0 - self.p3y) # Need to calculate turn angle (some relationship between theta0 theta1 and r)
        self.linear_distance = np.abs(self.p3y - y1) # Need to calculate linear distance from turn to radial turn (radial turn always at beginning/end of runup)
        
        m = (y1-self.p3y)/(x1-self.p3x) # slope between points 3 and 4
        self.radius = m * ((x0 - self.p3x) + (self.p3y - y0))/ (np.sqrt(m^2 + 1) - m) # radius of circle, can be subbed by variable r if known
    
        self.newpointx = x0 + self.radius - (m*(m*((x0+self.radius)-self.p3x) + (self.p3y-y0)))/(1+m^2)
        self.newpointy = y0+(m*((x0+self.radius)-self.p3x) + (self.p3y-y0))/(1+m^2)
        
        runup_length = np.sqrt((x1 - self.newpointx)^2 + (y1 - self.newpointy)^2)
    
        self.radial_distance = self.radius * (np.pi/2 + np.arctan(m)) # Need to calculate distance of radial turn based on r (distance as well as angle for turn)
        self.t_turn = self.omega_profile.total_time(self.theta_prime) 
        self.t_linear = self.accel_profile.total_time(self.linear_distance) 
        self.t_radial = self.accel_profile.total_time(self.radial_distance) 
        self.t_runup = self.accel_profile.total_time(runup_length)

    def step_enter_charger(self, t:float):
        '''
        phase:          turn, linear, radial turn, linear runup , no motion hold for setting, rotational hold
        control(flag): theta,    xy,       xy,       xy,               xy,                      theta
        '''
        ## ASSUSE FLAG=0 for theta, FLAG=1 for XY
        if (t < self.t_turn):
            current_theta = self.omega_profile.x(t, self.theta_prime)
            return self.x0, self.y0, current_theta, 0
        elif (t < self.t_turn + self.t_linear):
            changey = self.accel_profile.x(t - self.t1, self.linear_distance)
            return self.x0, self.y0 + changey, self.theta_prime, 1
        elif (t < self.t_turn + self.t_linear + self.t_radial):
            distance = self.accel_profile.x(t - self.t1 - self.t2, self.radial_distance)
            newy = self.y0 - self.radius*np.sin(distance/self.radius)
            newx =  self.x0 + self.radius(1-np.cos(distance/self.radius))
            return newx, newy, np.arctan2(newx - self.x0, newy - self.p3y), 1
        elif (t < self.t_turn + self.t_linear + self.t_radial + self.t_runup):
            changexy = self.accel_profile.x(t - self.t1 - self.t2 - self.t3, self.runup_length)
            return self.newpointx + changexy*np.cos(self.theta1), self.newpointy + changexy*np.sin(self.theta1), self.theta1, 1
        else:
            return self.x1, self.y1, self.theta1, 1 #holding
    

    def step_exit_charger(self, t:float):
        # if (t < self.t_runup):
        #     self.omega_profile.x(t, self.runup_length)
        #     return (x, y, theta, flag)
        # elif (t < self.t_runup + self.t_radial):
        #     self.accel_profile.x(t - self.t1, self.radial_distance)
        #     return (x, y, theta, flag)
        # elif (t < self.t_runup + self.t_radial + self.t_linear):
        #     self.accel_profile.x(t - self.t1 - self.t2, self.linear_distance)
        #     return (x, y, theta, flag)
        # elif (t < self.t_runup + self.t_radial + self.t_linear + self.t_turn):
        #     self.accel_profile.x(t - self.t1 - self.t2 - self.t3, self.theta_prime)
        #     return (x, y, theta, flag)
        # else:
        distance = self.accel_profile.x(t, self.runup_length)    
        return self.x0 + distance*np.cos(self.theta0), self.y0 + distance*np.sin(self.theta0), self.theta0, 1


    # 69: (0.0, 0.662, 0.325)

    def path_start(self):
        '''
        Desired position for navigation to start at after exiting charger
        '''
        return self.x1, self.y1

    def desired_path_end(self):
        '''
        Desired position for navigation to end at to begin entering charger
        '''
        return self.x0, self.y0