import numpy as np
import math
import matplotlib.pyplot as plt

from AccelerationLimitedProfile import AccelerationLimitedProile
#from Localization import InitialPosition

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

class EnterCharger:
    def __init__(self, x0:float, y0:float, theta0:float, x1:float, y1:float, theta1:float, runup_length:float, r:float, v_max:float, a_max:float, omega_max:float, alpha_max:float):
        self.accel_profile = AccelerationLimitedProile(v_max, a_max)
        self.omega_profile = AccelerationLimitedProile(omega_max, alpha_max)
        self.p1x, self.p1y, self.theta0, self.p5x, self.p5y, self.theta1, self.runup_length, self.r = x0, y0, theta0, x1, y1, theta1, runup_length, r

        #p1 = path/navigation starting/ending point (p1x, p1y, theta0)
        #p2 = point starting radial turn when coming into charger and ending radial turn when exiting charger (p2x, p2y)
        #p3 = point at center of circle (p3x, p3y)
        #p4 = point ending radial turn, starting runup coming into charger, or starting radial turn, after runup when leaving charger (p4x, p4y)
        #p5 = charger point (p5x, p5y, theta1)

        self.delta_theta = self.theta_in_negpi_pi(self.theta1 - self.theta0) # angle between theta0 and theta1
        self.p4x, self.p4y = self.p5x - self.runup_length * np.cos(self.theta1), self.p5y - self.runup_length * np.sin(self.theta1)
        self.p3x, self.p3y = self.find_p3() # center of circle
        self.p2x, self.p2y = self.find_p2() # point at which radial turn begins
        

        self.theta_prime = np.arctan2(self.p2y - self.p1y, self.p2x - self.p1x) # angle between p1 and p2
        self.linear_distance = np.sqrt((self.p2x - self.p1x)**2 + (self.p2y - self.p1y)**2) # distance between p1 and p2
        self.phi = self.theta_in_negpi_pi((np.arctan2((self.p4y - self.p3y)/self.r, (self.p4x - self.p3x)/self.r))-(np.arctan2((self.p2y - self.p3y)/self.r, (self.p2x - self.p3x)/self.r)))
        self.radial_distance = self.r * self.phi # distance of radial turn based on r

        print(f"point 1 {self.p1x, self.p1y, self.theta0}")
        print(f"point 2 {self.p2x, self.p2y}")
        print(f"point 3 {self.p3x, self.p3y}")
        print(f"point 4 {self.p4x, self.p4y}")
        print(f"point 5 {self.p5x, self.p5y, self.theta1}")

        self.t_turn = self.omega_profile.total_time(self.theta_prime - self.theta0) 
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
            return self.p1x, self.p1y, self.theta0 + self.omega_profile.x(t, self.theta_prime - self.theta0), 0
        elif (t < self.t_turn + self.t_linear):
            d = self.accel_profile.x(t - self.t_turn, self.linear_distance)
            return self.p1x + d * np.cos(self.theta_prime), self.p1y + d * np.sin(self.theta_prime), 0, 1
        elif (t < self.t_turn + self.t_linear + self.t_radial):
            distance = self.accel_profile.x(t - self.t_turn - self.t_linear, self.radial_distance)
            theta = distance/self.r
            theta_prime = np.arctan2((self.p2y - self.p3y)/self.r, (self.p2x - self.p3x)/self.r)
            change_x = self.p3x + self.r*np.cos(theta_prime + theta)
            change_y = self.p3y + self.r*np.sin(theta_prime + theta)
            return change_x, change_y, 0, 1
        elif (t < self.t_turn + self.t_linear + self.t_radial + self.t_runup):
            d = self.accel_profile.x(t - self.t_turn - self.t_linear - self.t_radial, self.runup_length)
            return self.p4x + d*np.cos(self.theta1), self.p4y + d*np.sin(self.theta1), 0, 1
        else:
            return self.p5x, self.p5y, self.theta1, 0 #holding position
    

    def theta_in_negpi_pi(self, theta):
        if theta > math.pi: 
            theta -= 2*math.pi
        if theta < -math.pi:
            theta += 2*math.pi
        return theta

    def find_p2(self):
        # Vector from P3 to P1
        vx, vy = self.p1x - self.p3x, self.p1y - self.p3y
        
        # Perpendicular unit vector (-vy, vx)
        length = np.hypot(vx, vy)  # Equivalent to sqrt(vx^2 + vy^2)
        ux, uy = -vy / length, vx / length  # Normalize

        # Compute both possible P2 points
        p2x1, p2y1 = self.p3x + self.r * ux, self.p3y + self.r * uy
        p2x2, p2y2 = self.p3x- self.r * ux, self.p3y - self.r * uy

        # Compute the angle between the P2s and P4
        angle1 = math.atan2(self.p4y - p2y1, self.p4x - p2x1)
        angle2 = math.atan2(self.p4y - p2y2, self.p4x - p2x2)

        # Choose the P2 that is closest to the desired angle
        if abs(angle1 - self.delta_theta) < abs(angle2 - self.delta_theta):
            return p2x1, p2y1
        else:
            return p2x2, p2y2
        

        # l = np.sqrt((self.p3x-self.p1x)**2 + (self.p3y-self.p1y)**2)  # Equivalent to sqrt(vx^2 + vy^2)
        # alpha = np.arccos(self.r / l)  # Angle between P1 and P3
        # beta = np.arctan2(self.p1x - self.p3x, self.p1y - self.p3y)  # Angle of the vector from P3 to P1

        # p2x = self.r * np.cos(alpha + beta)
        # p2y = self.r * np.sin(alpha + beta)

        # return p2x, p2y
        

    def find_p3(self):
        # Direction vector of the line
        vx, vy = self.p5x - self.p4x, self.p5y - self.p4y
        
        # Perpendicular unit vector (-vy, vx)
        length = np.hypot(vx, vy)  # Normalize
        ux, uy = -vy / length, vx / length

        # Compute both possible centers of the circle
        c1_x, c1_y = self.p4x + self.r * ux, self.p4y + self.r * uy
        c2_x, c2_y = self.p4x - self.r * ux, self.p4y - self.r * uy

        # Compute distances to P1
        d1 = np.hypot(c1_x - self.p1x, c1_y - self.p1y)
        d2 = np.hypot(c2_x - self.p1x, c2_y - self.p1y)

        # Choose the closer one
        return (c1_x, c1_y) if d1 < d2 else (c2_x, c2_y)
    


        
        # p3x1 = self.p4x + self.r/self.runup_length * (self.p5y - self.p4y)
        # p3y1 = self.p4y - self.r/self.runup_length * (self.p5x - self.p4x)
        # p3x2 = self.p4x - self.r/self.runup_length * (self.p5y - self.p4y)
        # p3y2 = self.p4y + self.r/self.runup_length * (self.p5x - self.p4x)
        # # Compute distances to P1
        # d1 = np.hypot(p3x1 - self.p1x, p3y1 - self.p1y)
        # d2 = np.hypot(p3x2 - self.p1x, p3y2 - self.p1y)

        # # Choose the closer one
        # return (p3x1, p3y2) if d1 < d2 else (p3x2, p3y2)
        

    def total_time(self):
        return self.t_turn + self.t_linear + self.t_radial + self.t_runup

    

if __name__ == '__main__':
    # Test EnterExitCharger

    enter_charger = EnterCharger(0, 1.5, np.pi/4, 1.5, 0, np.pi/4, 0.5, 0.25, 0.3, 0.5, 4, 8) 
    tf = enter_charger.total_time()
    n = 10000
    ts = np.linspace(0,tf,n)
    xs = np.zeros_like(ts)
    ys = np.zeros_like(ts)
    ths = np.zeros_like(ts)
    for i in range(n):
        x,y,theta, flag = enter_charger.step_enter_charger(ts[i])
        xs[i] = x
        ys[i] = y
        ths[i] = theta
    
    plt.figure(1)
    plt.plot(xs, ys, ".k")
    plt.axis("equal")

    # plt.figure(2)
    # plt.plot(ts, xs)

    # plt.figure(3)
    # plt.plot(ts, ys)

    # plt.figure(4)
    # plt.plot(ts, ths)
    plt.show()
    