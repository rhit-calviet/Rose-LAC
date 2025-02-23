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
        salp0 = 0.0001

        # Initial Angular Velocity [rad/s]
        w0 = np.zeros(3, dtype=np.float64)
        # Standard Deviation of initial angular velocity [rad/s]
        sw0 = 0.001

        # Initial Position
        x_pos0 = np.array([x0, y0, z0], dtype=np.float64)
        # Standard Deviation of initial position [m]
        sx0 = 0.0001

        # Initial Linear Velocity [m/s]
        v0 = np.zeros(3, dtype=np.float64)
        # Standard Deviation of inital velocity [m/s]
        sv0 = 0.0001

        # Initial Linear Acceleration [m/s]
        a0 = np.zeros(3, dtype=np.float64)
        # Standard Deviation of initial acceleration [m/s]
        sa0 = 0.001
        
        # Max expected angular acceleration [rad/s^2]
        w_dot_max = 4
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
        imu = Observations.IMUObservation(accel=accel, gyro=gyro, accel_var=0.01**2, gyro_var=0.01**2)
        odo = Observations.OdometryObservation(lin_speed, ang_speed, 0.05**2, 0.1**2, 0.1**2, 0.5**2)
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
        local_coords (numpy array): Array of positions in local frame [m] with shape (N, 3)
        var_local_coords (float): the variance in the local position measurement

        Returns:
        3 element numpy array: position in world frame [m]
        float: variance of world frame posiiton 
        """
        R = self.__mekf.rotation_matrix()
        world_coords = (R @ local_coords.T) + self.__mekf.position()[:,np.newaxis]

        local_coords_sqr_mag = np.sum(np.square(local_coords.T), axis=0)

        var_world = local_coords_sqr_mag * self.__mekf.orientation_variance() + self.__mekf.position_variance() + var_local_coords

        return world_coords.T, var_world
    
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
        local_coords = R.T @ (world_coords.T - self.__mekf.position()[:,np.newaxis])

        world_coords_sqr_mag = np.sum(np.square(local_coords), axis=0)

        var_local = world_coords_sqr_mag * self.__mekf.orientation_variance() + self.__mekf.position_variance() + var_world_coords

        return local_coords.T, var_local
    
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
        world_coords = R @ local_coords.T

        local_coords_sqr_mag = np.sum(np.square(local_coords.T), axis=0)

        var_world = local_coords_sqr_mag * self.__mekf.orientation_variance() + var_local_coords

        return world_coords.T, var_world
    
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
        local_coords = R.T @ world_coords.T

        world_coords_sqr_mag = np.sum(np.square(world_coords.T), axis=0)

        var_local = world_coords_sqr_mag * self.__mekf.orientation_variance() + var_world_coords

        return local_coords.T, var_local
    
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
    
    def current_angular_velocity(self) -> Tuple[np.ndarray, float]:
        """
        Get current robot angular velocity [rad] in world coordinates

        Returns:
        3 element numpy array: robot velocity in world frame
        float: variance of world frame velocity
        """
        return self.__mekf.angular_velocity(), self.__mekf.angular_velocity_variance()

    
    def current_2D_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
        """
        Get current robot 2D position [m] and orientation [rad] in world coordinates

        Returns:
        (x, y, theta), (position_variance, orientation_variance)

        x (float): current x position [m] in world coordinates
        y (float): current y position [m] in world coordinates
        theta (float): current rotation [rad] around z axis from world axis to local axis
        position_variance (float): the variance in the current position measurement [m]
        orientation_variance (float): the variance in the current orientation measurement [rad]
        """
        pos, pos_var = self.current_position()
        x_local = np.array([1,0,0])
        R = self.__mekf.rotation_matrix()
        x_world = R @ x_local

        x = pos[0]
        y = pos[1]
        theta = np.atan2(x_world[1], x_world[0])

        return (x, y, theta), (pos_var, self.__mekf.orientation_variance())

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
        v, v_var = self.current_velocity()
        w, w_var = self.current_angular_velocity()

        vx = v[0]
        vy = v[1]
        wz = w[3]
        return (vx, vy, wz), (v_var, w_var)

if __name__ == "__main__":
    robot = RobotPose(0,0,0,0,0,0)
    pos = np.random.rand(10,3)
    var = np.random.rand(10)
    wpos, wvar = robot.convert_world_to_local_position(pos,var)
    print(wpos)
    print()
    print(wvar)