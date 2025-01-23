import numpy as np
import MEKF
from typing import Tuple
import Observations

class RobotPose:
    def __init__(self, x0:float, y0:float, z0:float, roll0:float, pitch0:float, yaw0:float):
        # Sample Time
        dt = 1/20

        # Initial Rotation [rad]
        roll0 = roll0
        pitch0 = pitch0
        yaw0 = yaw0
        # Standard Deviation of initial rotation [rad]
        salp0 = 0

        # Initial Angular Velocity [rad/s]
        w0 = np.zeros(3, dtype=np.float64)
        # Standard Deviation of initial angular velocity [rad/s]
        sw0 = 0.001

        # Initial Position
        x_pos0 = np.array([x0, y0, z0], dtype=np.float64)
        # Standard Deviation of initial position [m]
        sx0 = 0.01

        # Initial Linear Velocity [m/s]
        v0 = np.zeros(3, dtype=np.float64)
        # Standard Deviation of inital velocity [m/s]
        sv0 = 0.0001

        # Initial Linear Acceleration [m/s]
        a0 = np.zeros(3, dtype=np.float64)
        # Standard Deviation of initial acceleration [m/s]
        sa0 = 0.001
        
        # Max expected angular acceleration [rad/s^2]
        w_dot_max = 2
        # Standard deviation of the equation w_dot = 0 [rad/s^2]
        sw = w_dot_max
        
        # Max expected jerk [m/s^3]
        a_dot_max = 4
        # Standard deviation of the equation a_dot = 0 [m/s^3]
        sa = a_dot_max
        
        # Lunar gravitational vector [m/s^2]
        g = np.array([0,0,-1.625], dtype=np.float64)

        self.__mekf = MEKF.MultiplicativeExtendedKalmanFilter(roll0, pitch0, yaw0, w0, x_pos0, v0, a0, salp0, sw0, sx0, sv0, sa0, sw, sa, dt, g)

        # Observations
        self.__points = []
        self.__directions = []

    def update(self, gyro:np.ndarray, accel:np.ndarray, lin_speed:float, ang_speed:float) -> None:
        """
        Update the state estimate with the new observations.
        This should be called once every update cycle.

        Parameters:
        Observations (Observations): the new observations

        Returns:
        None
        """

        # Construct observations
        imu = Observations.IMUObservation(accel=accel, gyro=gyro, accel_var=0.05, gyro_var=0.01)
        odo = Observations.OdometryObservation(lin_speed, ang_speed, 0.05, 1, 0.05, 3)
        observations = Observations.Observations(imu, odo, self.__points, self.__directions)

        # Update kalman filter with observtions
        self.__mekf.update(observations)

        # Clear all observations from this update
        self.__points = []
        self.__directions = []

    def add_point_observation(self, world_coord:np.ndarray, local_coord:np.ndarray, var:float) -> None:
        """
        Add an observation of a point
        
        Parameters:
        world_coord (numpy array): expected world coorinate position
        local_coord (numpy array): measured position in robot local coordinates
        var (float): variance in measurement
        """
        self.__points.append(Observations.PointObservation(world_coord, local_coord, var))

    def add_direction_observation(self, world_dir:np.ndarray, local_dir:np.ndarray, var:float) -> None:
        """
        Add an observation of a direction
        
        Parameters:
        world_dir (numpy array): expected world coorinate direction
        local_dir (numpy array): measured direction in robot local coordinates
        var (float): variance in measurement
        """
        self.__directions.append(Observations.DirectionObservation(world_dir, local_dir, var))

    def convert_local_to_world_position(self, local_coords:np.ndarray, var_local_coords:np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert position in local robot frame [x,y,z] to world coordinate frame [x,y,z]
        Also computes variance in world coordinate frame

        Parameters:
        local_coords (numpy array): 3 element array of [x,y,z] position in local frame [m]
        var_local_coords (float): the variance in the local position measurement

        Returns:
        3 element numpy array: position in world frame [m]
        float: variance of world frame posiiton 
        """
        R = self.__mekf.rotation_matrix()
        world_coords = R @ local_coords + self.__mekf.position()

        local_coords_sqr_mag = np.sum(np.square(local_coords), axis=1)

        var_world = local_coords_sqr_mag * self.__mekf.orientation_variance() + self.__mekf.position_variance() + var_local_coords

        return world_coords, var_world
    
    def convert_world_to_local_position(self, world_coords:np.ndarray, var_world_coords:float=0) -> Tuple[np.ndarray, float]:
        """
        Convert position in world coodinate frame [x,y,z] to local robot frame [x,y,z]
        Also computes variance in local coordinate frame

        Parameters:
        world_coords (numpy array): 3 element array of [x,y,z] position in world frame
        var_world_coords (float) (optional): the variance in the world position measurement, default 0

        Returns:
        3 element numpy array: position in local frame [m]
        float: variance of local frame position
        """
        R = self.__mekf.rotation_matrix()
        local_coords = R.T @ (world_coords - self.__mekf.position())

        world_coords_sqr_mag = np.sum(np.square(world_coords - self.__mekf.position()))

        var_local = world_coords_sqr_mag * self.__mekf.orientation_variance() + self.__mekf.position_variance() + var_world_coords

        return local_coords, var_local
    
    def convert_local_to_world_vector(self, local_coords:np.ndarray, var_local_coords:float=0) -> Tuple[np.ndarray, float]:
        """
        Convert vector in local robot frame [x,y,z] to world coordinate frame [x,y,z]
        Also computes variance in world coordinate frame

        Parameters:
        local_coords (numpy array): 3 element array of [x,y,z] coordinates in local frame
        var_local_coords (float) (optional): the variance in the local coordinate measurement, default 0

        Returns:
        3 element numpy array: coordinates in world frame
        float: variance of world frame coordinates
        """
        R = self.__mekf.rotation_matrix()
        world_coords = R @ local_coords

        local_coords_sqr_mag = np.sum(np.square(local_coords))

        var_world = local_coords_sqr_mag * self.__mekf.orientation_variance() + var_local_coords

        return world_coords, var_world
    
    def convert_world_to_local_vector(self, world_coords:np.ndarray, var_world_coords:float=0) -> Tuple[np.ndarray, float]:
        """
        Convert vector in world coodinate frame [x,y,z] to local robot frame [x,y,z]
        Also computes variance in local coordinate frame

        Parameters:
        world_coords (numpy array): 3 element array of [x,y,z] coordinates in world frame
        var_world_coords (float) (optional): the variance in the world coordinate measurement, default 0

        Returns:
        3 element numpy array: coordinates in local frame
        float: variance of local frame coordinates
        """
        R = self.__mekf.rotation_matrix()
        local_coords = R.T @ world_coords

        world_coords_sqr_mag = np.sum(np.square(world_coords))

        var_local = world_coords_sqr_mag * self.__mekf.orientation_variance() + var_world_coords

        return local_coords, var_local
    
    def current_position(self) -> Tuple[np.ndarray, float]:
        """
        Get current robot position [m] in world coordinates

        Returns:
        3 element numpy array: robot position in world frame
        float: variance of world frame position
        """
        return self.__mekf.position(), self.__mekf.position_variance()
    
    def current_velocity(self) -> Tuple[np.ndarray, float]:
        """
        Get current robot velocity [m] in world coordinates

        Returns:
        3 element numpy array: robot velocity in world frame
        float: variance of world frame velocity
        """
        return self.__mekf.velocity(), self.__mekf.velocity_variance()

    