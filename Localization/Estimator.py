import numpy as np
from RobotPose import RobotPose
from ElevationMap import ElevationMap
from typing import Tuple

class Estimator:
    def __init__(self, x0:float, y0:float, z0:float, roll0:float, pitch0:float, yaw0:float, map_size:float, cell_size:float, num_map_subcells:int, map_buffer:int, rock_var_threshold):
        self.robot = RobotPose(x0, y0, z0, roll0, pitch0, yaw0)
        x_min = -map_size / 2
        y_min = -map_size / 2
        num_cells = np.round(map_size/cell_size)
        self.map = ElevationMap(x_min, y_min, cell_size, num_cells, num_map_subcells, map_buffer)
        self.rock_var_threshold = rock_var_threshold

        self.points = None
        self.var = None

    def add_elevation_points(self, points:np.ndarray, variance:np.ndarray):
        """
        Update elevation map with points

        Parameters:
        points: (N,3) [x,y,z] points in 3D space
        var: (N) variance of each point measurement
        """
        self.points = np.concatenate((self.points, points), axis=0)
        self.var = np.concatenate((self.var, variance), axis=0)

    def add_point_observation(self, world_coord:np.ndarray, local_coord:np.ndarray, var:float) -> None:
        """
        Add an observation of a point
        
        Parameters:
        world_coord (numpy array): expected world coorinate position
        local_coord (numpy array): measured position in robot local coordinates
        var (float): variance in measurement
        """
        self.robot.add_point_observation(world_coord, local_coord, var)

    def add_direction_observation(self, world_dir:np.ndarray, local_dir:np.ndarray, var:float) -> None:
        """
        Add an observation of a direction
        
        Parameters:
        world_dir (numpy array): expected world coorinate direction
        local_dir (numpy array): measured direction in robot local coordinates
        var (float): variance in measurement
        """
        self.robot.add_direction_observation(world_dir, local_dir, var)

    def update(self, gyro:np.ndarray, accel:np.ndarray, lin_speed:float, ang_speed:float):
        """
        Update the robot state estimate and the elevation map with the new observations.
        This should be called once every update cycle.

        Parameters:
        Observations (Observations): the new observations

        Returns:
        None
        """
        self.robot.update(gyro, accel, lin_speed, ang_speed)
        points_world, var_world = self.robot.convert_local_to_world_position(self.points, self.var)
        self.map.update(points_world, var_world)
        
    def current_2D_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
        """
        Get current robot 2D position [m] and orientation [rad] in world coordinates

        Returns:
        (x, y, theta), (position_variance, orientation variance)

        x (float): current x position [m] in world coordinates
        y (float): current y position [m] in world coordinates
        theta (float): current rotation [rad] around z axis from world axis to local axis
        position_variance (float): the variance in the current position measurement [m]
        orientation_variance (float): the variance in the current orientation measurement [rad]
        """
        return self.robot.current_2D_pose()
        
    def current_2D_velocity(self) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
        """
        Get current robot 2D velocity [m/s] and angular velocity [rad/s] in world coordinates

        Returns:
        (vx, vy, wz), (v_variance, w_variance)

        vx (float): current x velocity [m/s] in world coordinates
        vy (float): current y velocity [m/s] in world coordinates
        wz (float): current angular velocity [rad/s] around world z axis
        v_variance (float): the variance in the current velocity measurement [m/s]
        w_variance (float): the variance in the current angular velocity measurement [rad/s]
        """
        return self.robot.current_2D_velocity()

    def get_cell_info(self, x_index:int, y_index:int, alpha:float=0.05) -> Tuple[float, float, float, float]:
        """
            Get information for elevation cell

            Parameters:
            x_index: (int) cell x index
            y_index: (int) cell y index
            rock_var_thresh: (float) variance threshold of a cell to detect a rock
            alpha: (float) Confidence interval p value

            Returns:
            elevation, rock, elevation uncertainty, rock uncertainty

            elevation (float): elevation of the cell [m]
            rock (boolean): True is the cell a rock
            elevation uncertainty (float): relative uncertainty in the elevation
            rock uncertainty (float): relative uncertainty in the rock estimate
        """
        return self.get_cell_info(x_index,y_index,self.rock_var_threshold, alpha)