import numpy as np
import matplotlib.pyplot as plt

class AccelerationLimitedProile:
    def __init__(self, v_max:float, a_max:float):
        self.v_max = v_max
        self.a_max = a_max

    def x(self, t:float, xf:float):
        sign = np.sign(xf)
        xf = abs(xf)
        if self.v_max*self.v_max/self.a_max < xf:
            ta = self.v_max / self.a_max
            tf = ta + xf / self.v_max
            if t < 0:
                return 0
            if t < ta:
                return 0.5*self.a_max*t*t
            if t < tf-ta:
                return 0.5*self.v_max*ta + self.v_max*(t-ta)
            if t < tf:
                return xf - 0.5*self.a_max*(tf-t)*(tf-t)
            return xf * sign
        ta = np.sqrt(xf/self.a_max)
        tf = 2*ta
        if t < 0:
            return 0
        if t < ta:
            return 0.5*self.a_max*t*t
        if t < tf:
            return xf - 0.5*self.a_max*(tf-t)*(tf-t)
        return xf * sign
    
    def total_time(self, xf:float):
        xf = abs(xf)
        if self.v_max*self.v_max/self.a_max < xf:
            ta = self.v_max / self.a_max
            tf = ta + xf / self.v_max
            return tf
        ta = np.sqrt(xf/self.a_max)
        tf = 2*ta
        return tf
        
        
    

if __name__ == "__main__":
    accel = AccelerationLimitedProile(0.4, 0.1)
    n = 1000
    t = np.linspace(0, 8, n)
    xs = np.zeros_like(t)
    xf = 1
    for i in range(n):
        xs[i] = accel.x(t[i], xf)
    plt.figure()
    plt.plot(t,xs)
    plt.show()