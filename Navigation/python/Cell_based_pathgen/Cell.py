import numpy as np
from scipy.special import fresnel
import matplotlib.pyplot as plt

class Cell:
    def __init__(self, cell_input:int, cell_output:int):
        """
        
           ___2___
          |       |
        3 |       | 1
          |_______|
              4
        """
        self.v = 1.6004735639
        self.a = 7.2710403810
        self.v_lin = 0.3
        self.cell_size = 0.75
        self.input = cell_input
        self.output = cell_output

    def position(self, t):
        if self.input == 1:
            if self.output == 3:
                # Straight
                return
            if self.output == 2:
                # Turn 90
                return
            if self.output == 4:
                # Turn 90
                return
            if self.output == 1:
                # Turn 180
                return
        if self.input == 2:
            if self.output == 4:
                # Straight
                return
            if self.output == 1:
                # Turn 90
                return
            if self.output == 3:
                # Turn 90
                return
            if self.output == 2:
                # Turn 180
                return
        if self.input == 3:
            if self.output == 1:
                # Straight
                return
            if self.output == 2:
                # Turn 90
                return
            if self.output == 4:
                # Turn 90
                return
            if self.output == 3:
                # Turn 180
                return
        if self.input == 4:
            return
    
    def __turn_xposition(self, t):
        ta = self.v / self.a
        tf = np.pi/(2*self.v) + self.v/self.a
        if t < 0:
            return 0
        if t < ta:
            fres_input = t*np.sqrt(self.a/np.pi)
            _,c = fresnel(fres_input)
            return np.sqrt(np.pi/self.a)*c
        if t < tf - ta:
            fres_input = self.v/np.sqrt(self.a*np.pi)
            _,c = fresnel(fres_input)
            return np.sqrt(np.pi/self.a)*c + (self.v_lin*(np.sin(- self.v*self.v/(2*self.a) + t*self.v) - np.sin(self.v*self.v/(2*self.a))))/self.v
        if t < tf:
            fres_input1 = self.v/(np.sqrt(self.a*np.pi))
            _, c1 = fresnel(fres_input1)
            fres_input2 = self.v/(np.sqrt(self.a)*np.sqrt(np.pi))
            s2, _ = fresnel(fres_input2)
            fres_input3 = (self.v*self.v - self.a*t*self.v + (self.a*np.pi)/2)/(np.sqrt(self.a)*self.v*np.sqrt(np.pi))
            s3, _ = fresnel(fres_input3)
            return (np.sqrt(np.pi)*c1)/np.sqrt(self.a) + (np.sqrt(2)*self.v_lin*np.cos(self.v*self.v/(2*self.a) + np.pi/4))/self.v + (self.v_lin*np.sqrt(np.pi)*(s2 - s3))/np.sqrt(self.a)
        return 0.375
        
    def __turn_yposition(self, t):
        return self.__turn_xposition(self.__turn_travel_time()-t)

    def __turn_travel_time(self):
        return np.pi/(2*self.v) + self.v/self.a
    

if __name__ == "__main__":
    cell = Cell()
    n = 10000
    ts = np.linspace(cell.travel_time(), n)
    x = np.zeros_like(ts)
    y = np.zeros_like(ts)
    for i in range(n):
        t = ts[i]
        x[i] = cell.xposition(t)
        y[i] = cell.yposition(t)

    plt.figure()
    plt.plot(x,y)

    plt.axis('equal')
    plt.show()
