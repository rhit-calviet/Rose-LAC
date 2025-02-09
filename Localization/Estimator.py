import numpy as np
from RobotPose import RobotPose
from ElevationMap import ElevationMap
from typing import Tuple

class Estimator:
    def __init__(self, x0:float, y0:float, z0:float, roll0:float, pitch0:float, yaw0:float):
        self.robot = RobotPose(x0, y0, z0, roll0, pitch0, yaw0)
        self.map = ElevationMap(-27/2, -27/2, 0.15, 180, 3, 4)

        self.points = None
        self.var = None

    def add_elevation_points(self, points:np.ndarray, variance:np.ndarray):
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
        
