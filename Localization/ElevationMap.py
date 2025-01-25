import numpy as np
from scipy.stats import chi2, t
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class ElevationMap:
    def __init__(self, x_min:float, y_min:float, cell_size:float, num_cells:int, num_subcells:int, buffer_size:float):
        self.__sub_cell_size = cell_size / num_subcells
        self.__num_buffer_cells = round(buffer_size / self.__sub_cell_size)
        self.__x_min = x_min - self.__num_buffer_cells * self.__sub_cell_size
        self.__y_min = y_min - self.__num_buffer_cells * self.__sub_cell_size
        self.__num_cells = num_cells
        self.__num_total_sub_cells = num_cells*num_subcells + 2*self.__num_buffer_cells
        self.__x_max = x_min + cell_size * self.__num_total_sub_cells
        self.__y_max = y_min + cell_size * self.__num_total_sub_cells

        self.__std_threshold = 10

        self.__mean_elevation = np.zeros((self.__num_total_sub_cells, self.__num_total_sub_cells))
        self.__var_elevation = np.full_like(self.__mean_elevation, np.inf)
        self.__weight_sum = np.zeros_like(self.__mean_elevation)
        self.__num_obs = np.zeros_like(self.__mean_elevation)
        self.__weight_sum2 = np.zeros_like(self.__mean_elevation)
        self.__inv_systematic_uncertainty = np.zeros_like(self.__mean_elevation)

    def update(self, points:np.ndarray, var:np.ndarray) -> None:
        """
        Update elevation map with points

        Parameters:
        points: (N,3) [x,y,z] points in 3D space
        var: (N) variance of each point measurement
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Remove indexes with too high variance or out of bounds
        valid_indexes = (var < self.__std_threshold*self.__std_threshold) & (x >= self.__x_min) & (x <= self.__x_max) & (y >= self.__y_min) & (y <= self.__y_max)
        x = x[valid_indexes]
        y = y[valid_indexes]
        z = z[valid_indexes]
        var = var[valid_indexes]

        # Calculate grid indices for each point
        x_indices = ((x - self.__x_min) // self.__sub_cell_size).astype(int)
        y_indices = ((y - self.__y_min) // self.__sub_cell_size).astype(int)

        # Create a flat index for the grid
        flat_indices = x_indices * self.__num_total_sub_cells + y_indices

        # Calculate weights
        weights = np.reciprocal(var)
        

        ####### Compute sample mean and standard deviation

        # Number of observations
        num_obs = np.bincount(flat_indices, minlength=self.__num_total_sub_cells * self.__num_total_sub_cells)

        # Total weight per cell
        weight_sum = np.bincount(flat_indices, weights=weights, minlength=self.__num_total_sub_cells * self.__num_total_sub_cells)
        weight_sum2 = np.bincount(flat_indices, weights=weights*weights, minlength=self.__num_total_sub_cells * self.__num_total_sub_cells)

        # Weighted sum of z values per cell
        weighted_sum = np.bincount(flat_indices, weights=weights * z, minlength=self.__num_total_sub_cells * self.__num_total_sub_cells)

        # Weighted sum of squared z values per cell
        weighted_sq_sum = np.bincount(flat_indices, weights=weights * z**2, minlength=self.__num_total_sub_cells * self.__num_total_sub_cells)

        # Compute means and variances
        mean = np.zeros(self.__num_total_sub_cells * self.__num_total_sub_cells)
        variance = np.zeros(self.__num_total_sub_cells * self.__num_total_sub_cells)
        non_empty = weight_sum > 0  # Cells with data
        mean[non_empty] = weighted_sum[non_empty] / weight_sum[non_empty]
        variance[non_empty] = (
            weighted_sq_sum[non_empty] / weight_sum[non_empty] - mean[non_empty] ** 2
        )

        # Reshape results into the grid
        mean = mean.reshape(self.__num_total_sub_cells, self.__num_total_sub_cells)
        variance = variance.reshape(self.__num_total_sub_cells, self.__num_total_sub_cells)
        weight_sum = weight_sum.reshape(self.__num_total_sub_cells, self.__num_total_sub_cells)
        weight_sum2 = weight_sum2.reshape(self.__num_total_sub_cells, self.__num_total_sub_cells)
        num_obs = num_obs.reshape(self.__num_total_sub_cells, self.__num_total_sub_cells)

        ####### Compute total mean and standard deviation

        # Update weights
        weight_new = self.__weight_sum + weight_sum
        nonzero_weights = weight_new > 0

        # Update mean
        mean_weighted_sum = self.__weight_sum * self.__mean_elevation + weight_sum * mean
        mean_new = np.divide(mean_weighted_sum, weight_new, out=np.zeros_like(mean_weighted_sum), where=nonzero_weights)

        # Update variance
        self_var_elev = self.__var_elevation
        self_var_elev[self_var_elev == np.inf] = 0
        var_weighted_sum = (self.__weight_sum * (self_var_elev + np.square(self.__mean_elevation - mean_new)) +
                                   weight_new * (     variance + np.square(mean - mean_new))
                           )
        var_new = np.divide(var_weighted_sum, weight_new, out=np.full_like(var_weighted_sum, np.inf), where=nonzero_weights)

        # Save updates
        self.__weight_sum = weight_new
        self.__mean_elevation = mean_new
        self.__var_elevation = var_new
        self.__weight_sum2 += weight_sum2
        self.__num_obs += num_obs

        ###### Compute systematic uncertainty

        # Minimum weight per cell
        min_weights = np.full(self.__num_total_sub_cells * self.__num_total_sub_cells, np.inf)
        np.minimum.at(min_weights, flat_indices, weights)
        min_weights[min_weights == np.inf] = 0  # Set to 0 for cells with no data

        min_weights = min_weights.reshape(self.__num_total_sub_cells, self.__num_total_sub_cells)

        self.__inv_systematic_uncertainty += min_weights

    def elevation_uncertainty(self) -> np.ndarray:
        return np.reciprocal(self.__inv_systematic_uncertainty, out=np.full_like(self.__inv_systematic_uncertainty, np.inf), where=self.__inv_systematic_uncertainty>0)
    
    def variance_uncertainty(self) -> np.ndarray:
        n_eff = np.divide(np.square(self.__weight_sum), self.__weight_sum2, out=np.zeros_like(self.__weight_sum), where=self.__weight_sum2 > 0)
        return 2 * np.divide(np.square(self.variance()), n_eff - 1, out=np.full_like(self.__weight_sum, np.inf), where=n_eff > 1)

    def elevation(self, alpha=0.05) -> np.ndarray:
        dof = self.__num_obs - 1
        dof[dof < 1] = 1
        # Standard Error of mean
        se_mean = np.sqrt(np.divide(self.__var_elevation, self.__weight_sum, out=np.zeros_like(self.__weight_sum), where=dof > 1))

        # Critical t-value
        t_critical = t.ppf(1 - alpha / 2, dof)
        # Confidence interval
        ci_lower = self.__mean_elevation - t_critical * se_mean
        ci_upper = self.__mean_elevation + t_critical * se_mean

        ci_lower[dof < 1] = -np.inf
        ci_upper[dof < 1] = np.inf

        return self.__mean_elevation, ci_lower, ci_upper
        
    def elevation_variance(self, alpha=0.05):
        dof = self.__num_obs - 1
        dof[dof < 1] = 1
        chi2_lower = chi2.ppf(alpha / 2, dof)
        chi2_upper = chi2.ppf(1 - alpha/2, dof)

        var = self.__var_elevation

        ci_lower = (dof * var) / chi2_upper
        ci_upper = (dof * var) / chi2_lower
        ci_lower[self.__num_obs < 1] = 0
        ci_upper[self.__num_obs < 1] = np.inf

        correction_factor = np.divide(self.__weight_sum - 1, self.__weight_sum, out=np.zeros_like(self.__weight_sum), where=self.__weight_sum > 0)
        pop_variance = np.multiply(correction_factor, self.__var_elevation, out=np.full_like(self.__var_elevation, np.inf), where=correction_factor > 0)

        return pop_variance, ci_lower, ci_upper


if __name__ == "__main__":
    min_dim = -27/2
    max_dim = 27/2
    w = 180

    size = 0.15

    m = ElevationMap(min_dim, min_dim, size, w, 1, 0)
    n = 1000000

    map_size = 40
    map_cell_size = 0.01
    map_cells = round(map_size/map_cell_size)
    map = np.random.rand(map_cells, map_cells) * 5
    map = gaussian_filter(map, sigma=50)
    map -= np.mean(map)
    map_var = np.random.rand(map_cells, map_cells) * 0.0001
    map_var = gaussian_filter(map_var, sigma=50)
    map_var[map_cells//2:(map_cells//2) + 30,map_cells//2:(map_cells//2) + 30] = 0.1

    for i in range(10):
        xs = (np.random.rand(n) - 0.5) * map_size
        ys = (np.random.rand(n) - 0.5) * map_size

        rows = ((xs + map_size/2) // map_cell_size).astype(int)
        cols = ((ys + map_size/2) // map_cell_size).astype(int)
        allowed = (rows >= 0) & (rows < map_cells) & (cols >= 0) & (cols < map_cells)
        rows = rows[allowed]
        cols = cols[allowed]
        xs = xs[allowed]
        ys = ys[allowed]

        zs = map[rows, cols]
        v = map_var[rows, cols]

        std_meas = np.sqrt(np.random.rand(rows.shape[0]) * 0.0001)

        xs += np.random.randn(rows.shape[0]) * std_meas
        ys += np.random.randn(rows.shape[0]) * std_meas
        zs += np.random.randn(rows.shape[0]) * np.sqrt(v) + np.random.randn(rows.shape[0]) * std_meas

        pts = np.array([xs, ys, zs])
        pts = pts.T

        t1 = time.time()
        m.update(pts,std_meas)
        dt = time.time() - t1
        print(dt)
    z, zlb, zub = m.elevation()
    zlb[zlb < -10] = 0
    zub[zub > 10] = 0
    v, vlb, vub = m.elevation_variance()
    v[v > 10] = 0
    vub[vub > 10] = 0

    plt.figure(1)
    plt.imshow(map, cmap="terrain")

    plt.figure(2)
    plt.imshow(map_var, cmap="terrain")

    plt.figure(3)
    plt.imshow(v, cmap="terrain")

    plt.figure(4)
    plt.imshow(vlb, cmap = "terrain")

    plt.figure(5)
    plt.imshow(vub, cmap="terrain")

    plt.figure(6)
    plt.imshow(z, cmap="terrain")

    plt.figure(7)
    plt.imshow(zlb, cmap = "terrain")

    plt.figure(8)
    plt.imshow(zub, cmap="terrain")

    plt.show()
    
