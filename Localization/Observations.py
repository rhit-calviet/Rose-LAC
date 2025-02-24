import numpy as np

class IMUObservation:
    def __init__(self, accel:np.ndarray, gyro:np.ndarray, accel_var:np.ndarray, gyro_var:np.ndarray):
        self.accel = np.squeeze(accel)
        self.accel_var = accel_var
        self.gyro = np.squeeze(gyro)
        self.gyro_var = gyro_var

class OdometryObservation:
    def __init__(self, lin_speed:float, ang_speed:float, lin_speed_var:float, lin_speed_off_axis_var:float, ang_speed_var:float, ang_speed_off_axis_var:float):
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed

        self.lin_speed_var = lin_speed_var
        self.lin_speed_off_axis_var = lin_speed_off_axis_var

        self.ang_speed_var = ang_speed_var
        self.ang_speed_off_axis_var = ang_speed_off_axis_var

class PointObservation:
    def __init__(self, world_coord:np.ndarray, local_coord:np.ndarray, var:float):
        self.world_coord = np.squeeze(world_coord)
        self.local_coord = np.squeeze(local_coord)
        self.var = var

class DirectionObservation:
    def __init__(self, world_dir:np.ndarray, local_dir:np.ndarray, var:float):
        self.world_dir = np.squeeze(world_dir)
        self.local_dir = np.squeeze(local_dir)
        self.var = var

class Observations:
    def __init__(self, imu:IMUObservation, odo:OdometryObservation, points:list[PointObservation], directions:list[DirectionObservation]):
        self.imu = imu
        self.odo = odo
        self.points = points
        self.directions = directions

