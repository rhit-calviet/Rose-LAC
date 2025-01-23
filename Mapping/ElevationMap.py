import numpy as np
from ConstantVariableEstimator import ConstantVariableEstimator

class ElevationMapCell:
    def __init__(self):
        self.elev = ConstantVariableEstimator()

    def update(self, elevation:float, var:float) -> None:
        self.elev.update(elevation, var)

    def elevation(self):
        return self.elev.mean()
    
    def var(self):
        return self.elev.variance()
    
class Rock:
    def __init__(self):
        self.x = ConstantVariableEstimator()
        self.y = ConstantVariableEstimator()
        self.z = ConstantVariableEstimator()
    
    def update(self, x:float, y:float, z:float, var:float):
        self.x.update(x,var)
        self.y.update(y,var)
        self.z.update(z,var)

    def position(self):
        return np.array([self.x.mean(), self.y.mean(), self.z.mean()])
    
    def variance(self):
        return self.x.variance() + self.y.variance() + self.z.variance()

class ElevationMap:
    def __init__(self, g_map:GeometricMap):
        map_length = g_map.get_cell_number()

        self.map:np.ndarray[ElevationMapCell] = np.empty((map_length, map_length), dtype=ElevationMapCell)

        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(0)
                g_map.set_cell_rock(i, j, False)
                self.map[i,j] = ElevationMapCell()
        self.g_map = g_map

        self.rocks = []


    def update_rock(self, x:float, y:float, z:float, var:float, thr:float) -> None:
        # Find the rock nearest to the measured point
        nearest_rock = None
        position = np.array([x,y,z])
        min_dist_sqr = float("inf")
        for rock in self.rocks:
            dist_sqr = np.sum(np.square(rock.position() - position))
            if dist_sqr < min_dist_sqr:
                nearest_rock = rock
                min_dist_sqr = dist_sqr
        
        # Determine if this rock is the same or new
        if nearest_rock.variance() + var > dist_sqr * thr * thr:
            # This is the same rock
            nearest_rock.update(x,y,z,var)
        else:
            # This is a new rock
            rock = Rock()
            rock.update(x,y,z,var)
            self.rocks.append(rock)


    def update(self, x:float, y:float, z:float, var:float) -> None:
        # Uncertainty threshold
        thr = 0.1 * self.g_map.get_cell_size()

        if var > thr*thr:
            return

        # Get cell
        i,j = self.g_map.get_cell_indexes(x, y)
        if i < 0 or i >= self.num_cells or j < 0 or j >= self.num_cells:
            return

        # Update Cell
        self.map[i,j].update_elevation(z, var)

    def update_geo_map(self):
        map_length = self.g_map.get_cell_number()

        mapping_tolerance = 0.1

        for i in range(map_length):
            for j in range(map_length):
                if self.map[i,j].variance() < mapping_tolerance:
                    self.g_map.set_cell_height(i,j,self.map[i,j].elevation())
                    self.g_map.set_cell_rock(i,j,False)
                else:
                    self.g_map.set_cell_height(i,j,float("inf"))
                    self.g_map.set_cell_rock(i,j,float("inf"))
        
        for rock in self.rocks:
            position = rock.position()
            self.g_map.set_rock(position[0], position[1], True)