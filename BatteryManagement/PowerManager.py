class PowerManager:
    def __init__(self, max_speed, initial_velocity, safety_margin=0.15):
        # Configurable component power consumption (in watts)
        self.camera_power = 3.0        # Per active camera
        self.light_power = 9.8         # Per active light
        self.wheel_power = 8.0         # Per active wheel
        self.computation_power = 8.0   # Constant computation load
        self.system_power = 2.0        # Base electronics consumption
        self.imu_power = 0.5           # Inertial measurement unit
        
        # Battery specifications (NASA-style Li-ion)
        self.battery_capacity = 283.0  # Wh (watt-hours)
        self.depth_of_discharge = 0.9  # 90% usable capacity
        
        # Operational parameters
        self.max_speed = max_speed     # m/s (maximum speed of the robot)
        self.velocity = initial_velocity  # current nominal velocity (m/s)
        self.safety_margin = safety_margin  # e.g., 15% extra energy buffer
        
        # Dynamic state
        self.current_power = self.battery_capacity * self.depth_of_discharge
        self.active_cameras = set()   # Tracking active camera IDs
        self.active_lights = set()    # Tracking active light IDs
        
        # Pre-calculate base consumption (always-on systems)
        self.base_consumption = self.computation_power + self.system_power + self.imu_power

    def update_power_state(self, dt, moving=False, distance_traveled=0):
        """
        Update power state based on active components and movement.
        dt: time delta in hours
        moving: whether wheels are active
        distance_traveled: distance in meters (used in degradation model)
        """
        # Calculate consumption from active components
        active_cam_power = len(self.active_cameras) * self.camera_power
        active_light_power = len(self.active_lights) * self.light_power
        wheel_power = 4 * self.wheel_power if moving else 0  # 4 wheels
        
        total_consumption = (self.base_consumption + active_cam_power + active_light_power + wheel_power) * dt
        
        # Apply a simple battery degradation model based on distance traveled
        degradation_factor = 1 - (0.000001 * distance_traveled)
        self.current_power = max(0, (self.current_power - total_consumption) * degradation_factor)

    def get_soc(self):
        """Return the current state of charge (SoC) as a percentage."""
        return (self.current_power / (self.battery_capacity * self.depth_of_discharge)) * 100

    def energy_required(self, distance, speed, use_cameras=0, use_lights=0):
        """
        Calculate the energy (in Wh) required for a maneuver.
        distance: in meters
        speed: in m/s (must be > 0 and <= max_speed)
        use_cameras: number of cameras expected to be used
        use_lights: number of lights expected to be used
        """
        if speed <= 0:
            raise ValueError("Speed must be positive")
        
        # Time in hours for the maneuver (converting seconds to hours)
        time_h = distance / speed / 3600
        
        # Assume wheels are active if moving
        wheel_usage = 4
        
        energy = (self.base_consumption +
                  use_cameras * self.camera_power +
                  use_lights * self.light_power +
                  wheel_usage * self.wheel_power) * time_h
        
        # Include a safety margin
        return energy * (1 + self.safety_margin)

    def should_return(self, distance_to_lander, current_speed, dynamic_load=0):
        """
        Determine if the robot should return to the lander.
        Returns a tuple: (should_return (bool), suggested_speed (m/s))
        distance_to_lander: meters
        current_speed: m/s
        dynamic_load: additional dynamic power load (if any) in watts
        """
        # Calculate minimum required energy including dynamic load
        min_energy = self.energy_required(
            distance_to_lander,
            current_speed,
            use_cameras=2,  # e.g., navigation cameras
            use_lights=2    # e.g., minimal lighting for return
        ) + dynamic_load
        
        # Determine an optimal return speed based on available energy
        suggested_speed = min(
            self.max_speed,
            current_speed * (self.current_power / min_energy)
        )
        
        # Adaptive safety threshold based on current SoC and mission area
        soc = self.get_soc()
        dynamic_threshold = min(25, 10 + (15 * (distance_to_lander / 27)))
        
        return (soc <= dynamic_threshold or self.current_power <= min_energy * 1.2, suggested_speed)
    
    def needs_to_return(self, distance_to_lander, current_speed, dynamic_load=0):
        """
        Returns True if the system determines that it needs to return to the lander to charge, otherwise False.
        """
        should_return_decision, _ = self.should_return(distance_to_lander, current_speed, dynamic_load)
        return should_return_decision
    
    # Component management utilities
    def enable_camera(self, camera_id):
        self.active_cameras.add(camera_id)

    def disable_camera(self, camera_id):
        self.active_cameras.discard(camera_id)

    def enable_light(self, light_id):
        self.active_lights.add(light_id)

    def disable_light(self, light_id):
        self.active_lights.discard(light_id)
