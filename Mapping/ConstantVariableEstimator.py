import numpy as np

class ConstantVariableEstimator:
    def __init__(self):
        self.xbar = 0
        self.sum_weight = 0
        self.S = 0

    def update(self, x:float, var:float) -> None:
        w = 1 / var
        self.sum_weight += w
        mean_old = self.xbar
        self.xbar += (w / self.sum_weight) * (x - mean_old)
        self.S += w * (x - mean_old) * (x - self.xbar)

    def variance(self) -> float:
        if self.sum_weight > 0:
            # Measured variance
            var_meas = self.S / self.sum_weight

            # Minimum possible variance based on measurements
            min_var_allowed = 1 / self.sum_weight
            
            return max(min_var_allowed, var_meas)
        return float("Inf")
    
    def mean(self) -> float:
        if self.sum_weight > 0:
            return self.xbar
        return 0