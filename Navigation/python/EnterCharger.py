import numpy as np

from Navigation.python.AccelerationLimitedProfile import AccelerationLimitedProile

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
        self.runup_length = runup_length
        self.theta_prime = False #TODO Need to calculate turn angle (some relationship between theta0 theta1 and r)
        self.linear_distance = False #TODO Need to calculate linear distance from turn to radial turn (radial turn always at beginning/end of runup)
        self.radial_distance = False #TODO Need to calculate distance of radial turn based on r (distance as well as angle for turn)
        self.t_turn = self.omega_profile.total_time(self.theta_prime) 
        self.t_linear = self.accel_profile.total_time(self.linear_distance) 
        self.t_radial = self.accel_profile.total_time(self.radial_distance) 
        self.t_runup = self.accel_profile.total_time(runup_length)

    def step_enter_charger(self, t:float):
        '''
        phase:          turn, linear, radial turn, linear runup , no motion hold for setting, rotational hold
        control(flag): theta,    xy,       xy,       xy,               xy,                      theta
        '''
        if (t < self.t_turn):
            self.omega_profile.x(t, self.theta_prime)
            return (x, y, theta, flag)
        elif (t < self.t_turn + self.t_linear):
            self.accel_profile.x(t - self.t1, self.linear_distance)
            return (x, y, theta, flag)
        elif (t < self.t_turn + self.t_linear + self.t_radial):
            self.accel_profile.x(t - self.t1 - self.t2, self.radial_distance)
            return (x, y, theta, flag)
        elif (t < self.t_turn + self.t_linear + self.t_radial + self.t_runup):
            self.accel_profile.x(t - self.t1 - self.t2 - self.t3, self.runup_length)
            return (x, y, theta, flag)
        else:
            return (x, y, theta, theta_flag)
    

    def step_exit_charger(self, t:float):
        if (t < self.t_runup):
            self.omega_profile.x(t, self.runup_length)
            return (x, y, theta, flag)
        elif (t < self.t_runup + self.t_radial):
            self.accel_profile.x(t - self.t1, self.radial_distance)
            return (x, y, theta, flag)
        elif (t < self.t_runup + self.t_radial + self.t_linear):
            self.accel_profile.x(t - self.t1 - self.t2, self.linear_distance)
            return (x, y, theta, flag)
        elif (t < self.t_runup + self.t_radial + self.t_linear + self.t_turn):
            self.accel_profile.x(t - self.t1 - self.t2 - self.t3, self.theta_prime)
            return (x, y, theta, flag)
        else:
            return (x, y, theta, theta_flag)


    def path_start():
        '''
        Desired position for navigation to start at after exiting charger
        '''
        return 0

    def desired_path_end():
        '''
        Desired position for navigation to end at to begin entering charger
        '''
        return 0