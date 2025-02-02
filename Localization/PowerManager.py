class PowerManager:
    def __init__(self, velocity):
       
        # Initialize the PowerManager with battery capacity, base consumption, max speed, and drive power.
        self.cameraConsumption = 3
        self.lightConsumption = 9.8
        self.wheelConsumption = 8
        self.battery_capacity = 283
        self.comptutationConsuption = 8 #constant
        self.nonConsumption = 2 #constant
        self.imuConsumption = 0.5 #constant
        self.base_consumption = self.comptutationConsuption + self.nonConsumption + self.imuConsumption + (8*self.cameraConsumption) + (8*self.lightConsumption)
        self.moving_consumption = self.wheelConsumption*4
        self.velocity = velocity

    def get_soc(self, current_power):

        # Calculate current SOC (%) based on remaining energy.
        remaining_energy = current_power
        return (remaining_energy / self.battery_capacity) * 100

    def energy_required_to_return(self, distance_to_lander, dynamic_load):

        # Calculate how much energy is needed to return to the lander.
        time_to_return = distance_to_lander / self.velocity  # Time in hours
        return (dynamic_load + self.base_consumption + self.moving_consumption) * time_to_return

    def should_return_to_lander(self, distance_to_lander, dynamic_load, current_power, basedOnDistance):
        
        soc = self.get_soc(current_power)

        if (basedOnDistance):

        # Energy needed to return to the lander
            energy_needed = self.energy_required_to_return(distance_to_lander, dynamic_load)
        
        else:
            energy_needed = self.energy_required_to_return(9, dynamic_load)

        # Compute dynamic threshold
        safe_return_threshold = ((energy_needed / self.battery_capacity) * 100) + 0.1

        return soc <= safe_return_threshold

        

