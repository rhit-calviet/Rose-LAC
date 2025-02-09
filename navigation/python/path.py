import numpy as np
import math
import heapq
from collections import deque
import matplotlib.pyplot as plt
import scipy.interpolate as si

### WORK TO BE DONE
### add position as a function of time functionality

class GeneratePath:

    path = deque()
    smooth_path = deque()

    def __init__(self, h, w, len, velocity):
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
    
    def set_lander(self):
        x_center, y_center = self.x //2, self.y //2
        self.set_rock(x_center, y_center)

    def set_rock(self, x, y):
        for i in range(-1, 2):
            for j in range(-1, 2):
                vertex = self.encode(x + i, y + j)
                for k in range(self.numVertices):
                    if self.matrix[k][vertex] != 0:
                        self.matrix[k][vertex] = 0

    def check_rock(self, x, y): ### needs debugging
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
    
    def update_path(self):
        x1, y1 = self.smooth_path[0]
        x1, y1 = round(x1), round(y1)
        x2, y2 = self.smooth_path[1]
        x2, y2 = round(x2), round(y2)
        if self.check_rock(x2, y2):
            print("Rock detected in path. Path is being updated...")
            index = 4
            x3, y3 = self.smooth_path[index]
            x3, y3 = round(x3), round(y3)
            while(self.check_rock(x3, y3)):
                index = index + 1
                x3, y3 = self.smooth_path[index]
                x3, y3 = round(x3), round(y3)
            start = self.encode(x1, y1)
            end = self.encode(x3, y3)
            new_path = self.astar(start, end)
            for _ in range(index + 20 + (self.num_points // 1000)):
                self.smooth_path.popleft()
            new_path = self.smooth_rover_path(new_path)
            self.smooth_path.extendleft(reversed(new_path))
            self.smooth_path = self.smooth_rover_path(self.smooth_path)
            print("Path update completed.")

    def smooth_rover_path(self, path_deque, smoothing_factor=15, num_points=10000):
        """
        Smooths a deque of (x, y) coordinate pairs using B-spline interpolation.

        :param path_deque: deque of (x, y) tuples representing the path
        :param smoothing_factor: controls the amount of smoothing (0 = exact fit, higher = smoother)
        :param num_points: number of points to sample along the smooth path
        :return: list of (x, y) smoothed coordinate pairs
        """

        if len(path_deque) < 4:
            print("Not enough points for spline interpolation. Skipping smoothing.")
            return

        # Convert deque to numpy arrays for processing
        path_array = np.array(list(path_deque))
        x, y = path_array[:, 0], path_array[:, 1]

        if len(x) != len(y):
            print(f"Mismatch in coordinate lengths: x={len(x)}, y={len(y)}")
            return

        if np.all(x == x[0]) or np.all(y == y[0]):
            print("All x or y values are the same. Cannot perform spline fitting.")
            return

        try:
            # Generate B-spline representation

            self.num_points = num_points

            tck, _ = si.splprep([x, y], s=smoothing_factor)

            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = si.splev(u_new, tck)

            return deque(list(zip(x_new, y_new)))
        
        except Exception as e:
            print(f"Error in smoothing path: {e}")

    def get_pos_at_time(self, time):
        distance = self.velocity * time
        num_vertices = int(distance / self.len)
        pos = self.path[num_vertices]
        return pos

    def next(self):
        if self.smooth_path[0]:
            (x, y) = self.smooth_path.popleft()
            self.update_path()
            return (x, y)
        else:
            print("No more points in path.")

    def visualize_path(self):
        """
        Visualizes the path on a grid.
        
        :param path: A deque containing (x, y) coordinates of the path.
        :param grid_size: A tuple (width, height) representing the grid dimensions.
        """

        path = self.smooth_path
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