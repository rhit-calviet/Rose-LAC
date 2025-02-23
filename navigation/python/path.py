import numpy as np
import math
import heapq
from itertools import islice
from collections import deque
import matplotlib.pyplot as plt
import scipy.interpolate as si
from scipy.integrate import quad

### WORK TO BE DONE
### add position as a function of time functionality

class GeneratePath:

    path = deque()
    smooth_path = deque()

    def __init__(self, h, w, len, velocity):
        self.step_size = 1.0
        self.cur_idx = 0
        self.y = int(h / len)
        self.x = int(w / len)
        self.len = len
        self.numVertices = self.x*self.y
        self.matrix = np.zeros((self.numVertices, self.numVertices))
        self.initialize_edges(len)
        self.generate_spiral_path()
        self.smooth_path = self.smooth_rover_path(self.path)
        self.set_lander()
        self.velocity = velocity

    def initialize_edges(self, len):
        for x in range(1, self.y - 1):
            for y in range(1, self.x - 1):
                hypot = math.sqrt(len**2 + len**2)

                encoded = self.encode(x, y)
                encoded_l = self.encode(x - 1, y)
                encoded_r = self.encode(x + 1, y)
                encoded_u = self.encode(x, y + 1)
                encoded_d = self.encode(x, y - 1)
                encoded_lu = self.encode(x - 1, y + 1)
                encoded_ru = self.encode(x + 1, y + 1)
                encoded_ld = self.encode(x - 1, y - 1)
                encoded_rd = self.encode(x + 1, y - 1)

                self.matrix[encoded][encoded_l] = len
                self.matrix[encoded][encoded_r] = len
                self.matrix[encoded][encoded_u] = len
                self.matrix[encoded][encoded_d] = len
                self.matrix[encoded][encoded_lu] = hypot
                self.matrix[encoded][encoded_ru] = hypot
                self.matrix[encoded][encoded_ld] = hypot
                self.matrix[encoded][encoded_rd] = hypot

        print("Finished initialization.")

    def encode(self, x, y):
        index = y * self.x + x
        return index

    def decode(self, vertexNum):
        x = vertexNum % self.x
        y = vertexNum // self.x
        return (x, y)
    
    def generate_spiral_path(self, lane_width=3, start_x=0, start_y=0):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        start_x, start_y = (self.x // 2) - 3, (self.y // 2) - 3

        path = deque()
        
        steps_in_layer = 6
        direction_index = 0
        steps_taken = 0
        layer_expansion = 0

        x, y = start_x, start_y
        
        while 0 <= x < self.x and 0 <= y < self.y:
            path.append((x, y))

            dx, dy = directions[direction_index]
            x, y = x + dx, y + dy
            steps_taken += 1

            if steps_taken == steps_in_layer:
                direction_index = (direction_index + 1) % 4
                steps_taken = 0
                layer_expansion += 1
                
                if layer_expansion % 2 == 0:
                    steps_in_layer += lane_width

        self.path = path
        print("Spiral path generated.")

    def get_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            verticeNum1 = self.encode(x1, y1)
            verticeNum2 = self.encode(x2, y2)
            distance = distance + self.matrix[verticeNum1][verticeNum2]
        return distance
    
    def set_lander(self): ## update this to account for dims of lander, rotation, and the charging area of the lander
        x_center, y_center = self.x //2, self.y //2
        self.set_rock(x_center, y_center, True)

    def set_rock(self, x, y, rock):
        if rock:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    vertex = self.encode(x + i, y + j)
                    for k in range(self.numVertices):
                        if self.matrix[k][vertex] != 0:
                            self.matrix[k][vertex] = 0
        else:
            hypot = math.sqrt(self.len**2 + self.len**2)

            encoded = self.encode(x, y)
            encoded_l = self.encode(x - 1, y)
            encoded_r = self.encode(x + 1, y)
            encoded_u = self.encode(x, y + 1)
            encoded_d = self.encode(x, y - 1)
            encoded_lu = self.encode(x - 1, y + 1)
            encoded_ru = self.encode(x + 1, y + 1)
            encoded_ld = self.encode(x - 1, y - 1)
            encoded_rd = self.encode(x + 1, y - 1)

            self.matrix[encoded][encoded_l] = len
            self.matrix[encoded][encoded_r] = len
            self.matrix[encoded][encoded_u] = len
            self.matrix[encoded][encoded_d] = len
            self.matrix[encoded][encoded_lu] = hypot
            self.matrix[encoded][encoded_ru] = hypot
            self.matrix[encoded][encoded_ld] = hypot
            self.matrix[encoded][encoded_rd] = hypot

    def check_rock(self, x, y):
        vertex = self.encode(x, y)
        for i in range(self.numVertices):
            if self.matrix[i][vertex] != 0:
                return False
        return True

    def get_neighbors(self, node):
        neighbors = []
        for neighbor, weight in enumerate(self.matrix[node]):
            if weight > 0:  # If there's a valid edge
                neighbors.append((neighbor, weight))
        return neighbors

    def heuristic(self, node, goal):
        x1, y1 = self.decode(node)
        x2, y2 = self.decode(goal)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def astar(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))  # (f-cost, node)
        
        g_costs = {node: float('inf') for node in range(self.numVertices)}
        g_costs[start] = 0

        came_from = {}  # Store paths

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(self.decode(current))
                    current = came_from[current]
                path.append(self.decode(start))
                return path[::-1]  # Reverse path

            for neighbor, edge_cost in self.get_neighbors(current):
                new_g_cost = g_costs[current] + edge_cost

                if new_g_cost < g_costs[neighbor]:  # Found a better path
                    g_costs[neighbor] = new_g_cost
                    f_cost = new_g_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_cost, neighbor))
                    came_from[neighbor] = current

        return None  # No path found
    
    def update_curr_idx(self, length):
        x1, y1 = self.get_cell_coords_at_length(length) ## gets edgy point at length

        x2, y2 = self.path[self.cur_idx] ## current point in edgy path
        dist_cur = (x2 - x1)**2 + (y2 - y1)**2 ## distance from current smooth point to the current edgy point

        x3, y3 = self.path[self.cur_idx + 1] ## next point in edgy path
        dist_new = (x3 - x1)**2 + (y3 - y1)**2 ## distance from current smooth point to the next edgy point

        while dist_new > dist_cur: ## while distance from current smooth point to current edgy point is greater than distance from next smooth point to current edgy point
            self.cur_idx += 1  ## update the edgy index to move us up one postition

            x2, y2 = self.path[self.cur_idx] ## get next edgy point
            dist_cur = (x2 - x1)**2 + (y2 - y1)**2 

            x3, y3 = self.path[self.cur_idx + 1]
            dist_new = (x3 - x1)**2 + (y3 - y1)**2

    def get_cell_coords_at_length(self, length):
        num_samples = 1000

        u_fine = np.linspace(0, 1, num_samples)
        x_fine, y_fine = si.splev(u_fine, self.tck)

        distances = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2)
        arc_lengths = np.concatenate(([0], np.cumsum(distances)))

        u_target = np.interp(length, arc_lengths, u_fine)
        x_target, y_target = si.splev(u_target, self.tck)

        return (x_target, y_target)
    
    def get_real_world_coords(self, x_curr, y_curr):
        x_curr -= 0.5 * self.x
        y_curr -= 0.5 * self.y

        x_curr *= self.len
        y_curr *= self.len

        return x_curr, y_curr
        
    def update(self, length, display=False):
        # print(f'DEBUG: length-{length}, display-{display}')
        self.update_curr_idx(length)
        curr_smooth_idx = self.cur_idx
        begin_idx = math.ceil(length / self.step_size)

        final_path = deque(islice(self.smooth_path, begin_idx)) ## GOOD

        # self.visualize_path(final_path)

        while curr_smooth_idx + 1 < len(self.path):
            x_s, y_s = self.path[curr_smooth_idx]
            x_n, y_n = self.path[curr_smooth_idx + 1]

            # start_idx = curr_smooth_idx
            start = self.encode(x_s, y_s)

            while self.check_rock(x_n, y_n):
                # print(f'DEBUG: index-{curr_smooth_idx}, rock found at-{x_n, y_n}')
                curr_smooth_idx += 1
                x_n, y_n = self.path[curr_smooth_idx + 1]
                # print(f'DEBUG: last position checked for rock-{x_n, y_n}')

            x_l, y_l = self.path[curr_smooth_idx]

            # print(f'DEBUG: checking if is rock-{x_l, y_l}')
            
            curr_smooth_idx += 1

            # print(f'DEBUG: start idx-{curr_smooth_idx}, start rock-{x_s, y_s}, next rock-{x_n, y_n}')
            if self.check_rock(x_l, y_l): ## ERROR: doesn't go through this portion
                print(f'DEBUG: continue through') 
                x_e, y_e = self.path[curr_smooth_idx]
                end_idx = curr_smooth_idx
                end = self.encode(x_e, y_e)


                # front_of_path = deque(islice(self.path, start_idx))

                back_of_path = deque(islice(self.path, end_idx, len(self.path)))

                new_path = self.astar(start, end)

                if new_path:
                    # front_of_path.extend(new_path) ## USE ARC LENGTH OF THIS TO GET COORD
                    # new_length = self.approximate_arc_length(front_of_path)
                    # start_idx = math.ceil(new_length / self.step_size)
                    # print(f'DEBUG: start_idx was {start_idx}, reset start_idx to {end_idx}')
                    # start_idx = end_idx
                    # self.visualize_path(front_of_path)
                    # front_of_path.extend(back_of_path)
                    new_front = deque(islice(self.path, end_idx - len(new_path)))
                    # self.visualize_path(new_front)
                    new_path.extend(back_of_path)
                    # self.visualize_path(new_path)
                    # print(f'DEBUG: final path-{front_of_path}')
                    # final_path = self.smooth_rover_path(new_path)
                    # self.visualize_path(final_path)
                    new_front.extend(new_path)
                    self.path = new_front
                    self.smooth_path = self.smooth_rover_path(self.path)
            
        if display:
            self.visualize_path(self.smooth_path)
                    
        ## this portion of the code needs to be completed to take the portion of the path at and before curr_smooth_index, and append it to the front of the newly computed remainder of the path.

        # print(f'DEBUG: new_path', final_path)
        # original_front_path = deque(islice(self.smooth_path, start_idx))
        # if len(original_front_path):
        #     self.visualize_path(original_front_path)
        #     self.visualize_path(final_path)
        # print(f'DEBUG: OG_front_path - {start_idx}')
        # print(f'DEBUG: OG_front_path - {original_front_path}')
        # original_front_path.extend(final_path) ## ERROR HERE: not getting populated

        # print(f'DEBUG: OG_front_path2 - {original_front_path}')
        # print(f'DEBUG: smooth_path - {self.smooth_path}')

        # self.smooth_path = original_front_path

        # if display:
            # self.visualize_path(self.smooth_path)
        # .extend(final_path)
        # .extend(final_path)

        # x1, y1 = self.smooth_path[self.cur_idx]
        # x1, y1 = round(x1), round(y1)
        # x2, y2 = self.smooth_path[self.cur_idx + 1]
        # x2, y2 = round(x2), round(y2)
        # if self.check_rock(x2, y2):
        #     print("Rock detected in path. Path is being updated...")
        #     index = 4
        #     x3, y3 = self.smooth_path[index]
        #     x3, y3 = round(x3), round(y3)
        #     while(self.check_rock(x3, y3)):
        #         index = index + 1
        #         x3, y3 = self.smooth_path[index]
        #         x3, y3 = round(x3), round(y3)
        #     start = self.encode(x1, y1)
        #     end = self.encode(x3, y3)
        #     new_path = self.astar(start, end)
        #     for _ in range(index):
        #         self.smooth_path.popleft()
        #     new_path = self.smooth_rover_path(new_path)
        #     self.smooth_path.extendleft(reversed(new_path))
        #     self.smooth_path = self.smooth_rover_path(self.smooth_path)
        #     print("Path update completed.")

    def approximate_arc_length(self, path_deque):
        """Assume we're being passed a smooth path"""

        distance_per_step = self.step_size * 0.15
        
        return len(path_deque) * distance_per_step

    def compute_arc_length(self, path_deque, smoothing_factor=15):
        """
        Computes the exact arc length of a given path using B-spline representation 
        and numerical integration.

        :param path_deque: deque of (x, y) tuples representing the path
        :param smoothing_factor: smoothing parameter for B-spline
        :return: Total arc length of the path
        """

        if len(path_deque) < 4:
            print("Not enough points for spline fitting. Returning Euclidean length instead.")
            return sum(np.linalg.norm(np.array(path_deque[i]) - np.array(path_deque[i-1])) for i in range(1, len(path_deque)))

        path_array = np.array(list(path_deque))
        x, y = path_array[:, 0], path_array[:, 1]

        try:
            # Fit B-spline
            tck, _ = si.splprep([x, y], s=smoothing_factor)

            # Compute derivatives dx/du and dy/du
            def integrand(u):
                dx_du, dy_du = si.splev(u, tck, der=1)
                return np.sqrt(dx_du**2 + dy_du**2)  # Arc length formula

            # Perform numerical integration over [0,1]
            arc_length, _ = quad(integrand, 0, 1, limit=1000)

            return arc_length

        except Exception as e:
            print(f"Error computing arc length: {e}")
            return 0.0


    def smooth_rover_path(self, path_deque, smoothing_factor=15, step_size=1.0):
        """
        Smooths a deque of (x, y) coordinate pairs using B-spline interpolation 
        and reparameterizes the curve to maintain a constant velocity with a 
        user-defined step size.

        :param path_deque: deque of (x, y) tuples representing the path
        :param smoothing_factor: controls the amount of smoothing (0 = exact fit, higher = smoother)
        :param step_size: desired constant step size (distance between consecutive points)
        :return: deque of (x, y) smoothed coordinate pairs with constant step size
        """

        # print(f'DEBUG: smoothing path-{len(path_deque)}')

        path_deque = self.remove_duplicate_consecutive_points(path_deque)

        self.step_size = step_size

        if len(path_deque) < 4:
            print("Not enough points for spline interpolation. Skipping smoothing.")
            return path_deque

        # Convert deque to numpy arrays for processing
        path_array = np.array(list(path_deque))
        x, y = path_array[:, 0], path_array[:, 1]

        try:
            # Generate B-spline representation
            tck, u = si.splprep([x, y], s=smoothing_factor)
            self.tck = tck
            self.u = u

            # Compute arc lengths
            u_fine = np.linspace(0, 1, len(x) * 10)  # Fine sampling for arc length approximation
            x_fine, y_fine = si.splev(u_fine, tck)
            distances = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2)
            cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)  # Arc length parameterization

            # Define the new arc length values at uniform intervals with step_size
            total_length = cumulative_lengths[-1]
            num_points = int(total_length / step_size)  # Calculate required points based on step size
            uniform_lengths = np.linspace(0, total_length, num_points)

            # Resample at uniform arc length intervals
            u_new = np.interp(uniform_lengths, cumulative_lengths, u_fine)  # Map arc length to parameter space
            x_new, y_new = si.splev(u_new, tck)

            return deque(list(zip(x_new, y_new)))

        except Exception as e:
            print(f"Error in smoothing path: {e}")
            return path_deque
        
    def remove_duplicate_consecutive_points(self, path_deque):
        """Removes consecutive duplicate points in a deque"""
        unique_points = [path_deque[0]]
        for i in range(1, len(path_deque)):
            if path_deque[i] != path_deque[i - 1]:  # Skip duplicates
                unique_points.append(path_deque[i])
        return deque(unique_points)

    # def next(self, time):
    #     if self.smooth_path[0]:
    #         (x, y) = self.smooth_path[time]
    #         return (x, y)
    #     else:
    #         print("No more points in path.")

    def visualize_path(self, path=None):
        """
        Visualizes the path on a grid.
        
        :param path: A deque containing (x, y) coordinates of the path.
        :param grid_size: A tuple (width, height) representing the grid dimensions.
        """

        # path = self.smooth_path
        grid_size = (self.x, self.y)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1, grid_size[0])
        ax.set_ylim(-1, grid_size[1])
        ax.set_xticks(range(grid_size[0]))
        ax.set_yticks(range(grid_size[1]))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        # Extract x and y coordinates from path
        x_coords, y_coords = zip(*path)

        # Plot the path
        ax.plot(x_coords, y_coords, marker="o", markersize=4, linestyle="-", color="blue", label="Path")

        # Mark the start and end points
        ax.scatter(x_coords[0], y_coords[0], color="green", s=100, label="Start")
        ax.scatter(x_coords[-1], y_coords[-1], color="red", s=100, label="End")

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Path Visualization")
        ax.legend()
        
        plt.show()
