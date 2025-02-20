#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""
import time
import json
import math
from numpy import random
import numpy as np
from PIL import Image
from Localization.Estimator import Estimator
from Localization.Controller import Controller
from Localization.CameraFrameTransform import CameraFrameTransform
from Navigation.python.path import GeneratePath
from Navigation.python.AccelerationLimitedProfile import AccelerationLimitedProile
from perception.python.LocalCoordinates import LocalCoordinates
from perception.python.depth import DepthMap

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent


def get_entry_point():
    return 'TestAgent'


class TestAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        tf0 = self.get_initial_position()

        self.geomap = self.get_geometric_map()
        
        map_size = self.geomap.get_map_size()
        cell_size = self.geomap.get_cell_size()

        self.estimator = Estimator(tf0.location.x, tf0.location.y, tf0.location.z, tf0.rotation.pitch, tf0.rotation.roll, tf0.rotation.yaw, map_size, cell_size, num_map_subcells=2, map_buffer=4)
        self.controller = Controller(dt=0.05, v_min=-0.35, v_max=0.35, w_max=3, zeta_v=2, wn_v=2.5, zeta_w=2, wn_w=2.5)
        self.accel_profile = AccelerationLimitedProile(v_max=0.3, a_max=0.5)
        self.camera_transformer = CameraFrameTransform()
        self.depth_generator = DepthMap(img_size=(2248, 2048), baseline=0.162, fx=1751.28, fy=1465.12, cx=1224, cy=1024)
        self.path_generator = GeneratePath(map_size, map_size, cell_size, velocity=0.2)

        self.cameras = [carla.SensorPosition.Front,
                        carla.SensorPosition.FrontLeft,
                        carla.SensorPosition.FrontRight,
                        carla.SensorPosition.Left,
                        carla.SensorPosition.Right,
                        carla.SensorPosition.BackLeft,
                        carla.SensorPosition.BackRight,
                        carla.SensorPosition.Back]

        self.set_front_arm_angle(np.pi/3)
        self.set_back_arm_angle(np.pi/3)

    def use_fiducials(self):
        return True

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 0, 'width': '2248', 'height': '2048'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1, 'width': '2248', 'height': '2048'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 1, 'width': '2248', 'height': '2048'
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': '2248', 'height': '2048'
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': '2248', 'height': '2048'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': '2248', 'height': '2048'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '2248', 'height': '2048'
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 0, 'width': '2248', 'height': '2048'
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""

        mission_time = self.get_mission_time()

        front_left = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
        front_right = input_data['Grayscale'][carla.SensorPosition.FrontRight]

        self.depth_generator.compute(front_left, front_right)

        # TODO: Add elevation observations
        self.estimator.add_elevation_points(self.depth_generator.vectorized_positions, self.depth_generator.vectorized_variances)

        # Check for fiducials
        for cam in self.cameras:
            img = input_data['Grayscale'][cam]
            if img is not None: # None if no image taken
                local_coordinates = LocalCoordinates(img) 
                point_observation = local_coordinates.get_coordinates() # Returns None if no fiducials are detected
                if point_observation is not None:
                    for point in point_observation:
                        world_coord, local_coord, var = point
                        self.estimator.add_point_observation(world_coord, self.camera_transformer.carla_object_camera_to_robot_frame(local_coord, cam), var)

        # TODO: Add Direction observations
        #self.estimator.add_direction_observation()
            
        # Get sensor measurements
        lin_speed = self.get_linear_speed()
        ang_speed = self.get_angular_speed()
        imu = self.get_imu_data()

        gyro = np.array(imu[4:])
        accel = np.array(imu[:4])

        # Perform estimator update
        self.estimator.update(gyro, accel, lin_speed, ang_speed)

        # Get Current Position
        (x_curr, y_curr, theta_curr), (var_pos, var_ang) = self.estimator.current_2D_pose()
        (xdot_curr, ydot_curr, thetadot_curr), (var_vel, var_ang_vel) = self.estimator.current_2D_velocity()

        # TODO: Compute Navigation
        elev, rock, elev_uncert, rock_uncert = self.estimator.get_cell_info(x_index, y_index)
        x_desired = 0
        y_desired = 0
        theta_desired = 0
        curr_path_arclength = self.accel_profile.x(mission_time, total_arc_length)

        # Compute Control Inputs
        v,w = self.controller.compute_control_inputs(x_curr, y_curr, theta_curr, xdot_curr, ydot_curr, thetadot_curr, x_desired, y_desired, theta_desired, angle_control=False)
        control = carla.VehicleVelocityControl(v, w)

        # Determine which camera to turn on to track lander
        best_alignment = -1
        best_cam = None
        for cam in self.cameras:
            x_axis = np.ndarray([[1],
                                 [0],
                                 [0]])
            robot_frame_cam_x_axis = self.camera_transformer.carla_object_camera_to_robot_frame(x_axis, cam)
            world_cam_x_axis, _ = self.estimator.robot.convert_local_to_world_vector(robot_frame_cam_x_axis.T, 0)
            world_robot_pos, _ = self.estimator.robot.current_position()
            alignment = -np.dot(world_robot_pos, world_cam_x_axis)
            if alignment > best_alignment:
                best_alignment = alignment
                best_cam = cam
        # Set cameras
        for cam in self.cameras:
            if cam in [carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight, best_cam]:
                self.set_camera_state(cam, True)
                self.set_light_state(cam, 1)
            else:
                self.set_camera_state(cam, False)
                self.set_light_state(cam, 0)

        if mission_time > 10000:
            self.mission_complete()

        return control

    def finalize(self):
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                elev, rock, elev_uncert, rock_uncert = self.estimator.get_cell_info(i,j)
                g_map.set_cell_height(elev)
                g_map.set_cell_rock(i, j, rock)
