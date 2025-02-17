import numpy as np

import carla
"""
X-axis: Camera Out-Of-Plane
Y-axis: Camera Horizontal
Z-axis: Camera Vertical
"""

class CameraFrameTransform:
    def __init__(self):
        c30 = np.cos(np.deg2rad(30))
        s30 = np.sin(np.deg2rad(30))

        self.CAMERA_TRANSFORMATION_MATRICES = {
            'front left':        np.array([[ 1,  0,  0,  0.280],
                                           [ 0,  1,  0,  0.081],
                                           [ 0,  0,  1,  0.131],
                                           [ 0,  0,  0,  1    ]]),
            'front right':       np.array([[ 1,  0,  0,  0.280],
                                           [ 0,  1,  0, -0.081],
                                           [ 0,  0,  1,  0.131],
                                           [ 0,  0,  0,  1    ]]),
            'back left':         np.array([[-1,  0,  0, -0.280],
                                           [ 0, -1,  0, -0.081],
                                           [ 0,  0,  1,  0.131],
                                           [ 0,  0,  0,  1    ]]),
            'back right':        np.array([[-1,  0,  0, -0.280],
                                           [ 0, -1,  0,  0.081],
                                           [ 0,  0,  1,  0.131],
                                           [ 0,  0,  0,  1    ]]),
            'left':              np.array([[ 0, -1,  0,  0.015],
                                           [ 1,  0,  0,  0.252],
                                           [ 0,  0,  1,  0.132],
                                           [ 0,  0,  0,  1    ]]),
            'right':             np.array([[ 0,  1,  0, -0.015],
                                           [-1,  0,  0, -0.252],
                                           [ 0,  0,  1,  0.132],
                                           [ 0,  0,  0,  1    ]]),
            'front arm center':  np.array([[ 1,  0,  0,  0.222],
                                           [ 0,  1,  0,  0.000],
                                           [ 0,  0,  1,  0.061],
                                           [ 0,  0,  0,  1    ]]),
            'rear arm center':   np.array([[ 1,  0,  0, -0.223],
                                           [ 0,  1,  0,  0.000],
                                           [ 0,  0,  1,  0.061],
                                           [ 0,  0,  0,  1    ]]),
            'front arm':         np.array([[ 0,  0,  1,  0.414],
                                           [ 0,  1,  0,  0.015],
                                           [-1,  0,  0, -0.038],
                                           [ 0,  0,  0,  1    ]]),
            'rear arm':          np.array([[c30, 0,-s30,-0.339],
                                           [ 0,  1,  0,  0.017],
                                           [s30, 0, c30, 0.083],
                                           [ 0,  0,  0,  1    ]]),
        }

        self.body_fixed_cameras = [ carla.SensorPosition.FrontLeft,
                                    carla.SensorPosition.FrontRight,
                                    carla.SensorPosition.Left,
                                    carla.SensorPosition.Right,
                                    carla.SensorPosition.BackLeft,
                                    carla.SensorPosition.BackRight]
        self.front_arm_angle = 0
        self.rear_arm_angle = 0

    def update_arm_angles(self, front_arm_angle:float, rear_arm_angle:float):
        self.front_arm_angle = front_arm_angle
        self.rear_arm_angle = rear_arm_angle

    def carla_object_camera_to_robot_frame(self, camera_coords:np.ndarray, camera):
        """
        Camera coords: (3,N) array of coordinates in camera frame
        x-axis: camera out-of-plane axis (points in direction of camera)
        y-axis: camera horizontal axis (points in direction of left of picture)
        z-axis: camera vertical axis (points in direction of top of picture)
        """
        carla2name = {  carla.SensorPosition.Front:         'front arm',
                        carla.SensorPosition.FrontLeft:     'front left',
                        carla.SensorPosition.FrontRight:    'front right',
                        carla.SensorPosition.Left:          'left',
                        carla.SensorPosition.Right:         'right',
                        carla.SensorPosition.BackLeft:      'back left',
                        carla.SensorPosition.BackRight:     'back right',
                        carla.SensorPosition.Back:          'back'}
        
        return self.camera_to_robot_frame(camera_coords, carla2name[camera])


    def camera_to_robot_frame(self, camera_coords:np.ndarray, camera:str):
        """
        Camera coords: (3,N) array of coordinates in camera frame
        x-axis: camera out-of-plane axis (points in direction of camera)
        y-axis: camera horizontal axis (points in direction of left of picture)
        z-axis: camera vertical axis (points in direction of top of picture)

        camera names:
        """
        camera = camera.lower()
        camera_coords = np.vstack([camera_coords, np.ones(camera_coords.shape[1])])
        T = np.eye(4)
        if camera in self.body_fixed_cameras:
            T = self.CAMERA_TRANSFORMATION_MATRICES[camera]
        elif camera == 'front arm':
            T01 = self.CAMERA_TRANSFORMATION_MATRICES['front arm center']
            T23 = self.CAMERA_TRANSFORMATION_MATRICES['front arm']
            c = np.cos(self.front_arm_angle)
            s = np.sin(self.front_arm_angle)
            T12 = np.array([[ c,  0, -s,  0],
                            [ 0,  1,  0,  0],
                            [ s,  0,  c,  0],
                            [ 0,  0,  0,  1]])
            T = T01 @ T12 @ T23
        elif camera == 'rear arm':
            T01 = self.CAMERA_TRANSFORMATION_MATRICES['rear arm center']
            T23 = self.CAMERA_TRANSFORMATION_MATRICES['rear arm']
            c = np.cos(-self.rear_arm_angle)
            s = np.sin(-self.rear_arm_angle)
            T12 = np.array([[ c,  0, -s,  0],
                            [ 0,  1,  0,  0],
                            [ s,  0,  c,  0],
                            [ 0,  0,  0,  1]])
            T = T01 @ T12 @ T23
        else:
            print("Camera frame transform name not found")
        
        robot_coords = T @ camera_coords
        robot_coords = robot_coords[1:3,:]
        return robot_coords
