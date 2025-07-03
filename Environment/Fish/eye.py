import numpy as np
import math
import matplotlib.pyplot as plt
#import cupy as cp
from skimage.draw import circle_perimeter

def ray_sum_fully_vectorized(array, start_coord, angles_rad, step_size=0.5):
    """
    Fully vectorized version with NO loops - handles duplicate removal vectorized.
    
    Parameters:
    -----------
    array : numpy.ndarray
        2D numpy array containing pixel values
    start_coord : tuple
        Starting coordinate (row, col)
    angles_rad : array-like
        Array of angles in radians
    step_size : float
        Ray sampling step size (smaller = more accurate but slower)
        
    Returns:
    --------
    numpy.ndarray
        Sum of pixel values along each ray
    """
    angles_rad = np.atleast_1d(angles_rad)
    n_angles = len(angles_rad)
    
    if array.ndim != 2:
        raise ValueError("Array must be 2D")
    
    rows, cols = array.shape
    start_row, start_col = start_coord
    
    if not (0 <= start_row < rows and 0 <= start_col < cols):
        return np.zeros(n_angles)
    
    # Direction vectors
    dy = np.sin(angles_rad)  # Shape: (n_angles,)
    dx = np.cos(angles_rad)   # Shape: (n_angles,)
    
    # Calculate ray boundary intersections for each angle
    steps_to_right = np.where(dx > 0, (cols - 1 - start_col) / dx,
                             np.where(dx < 0, -start_col / dx, np.inf))
    steps_to_bottom = np.where(dy > 0, (rows - 1 - start_row) / dy,
                              np.where(dy < 0, -start_row / dy, np.inf))
    max_steps_per_angle = np.maximum(0, np.minimum(steps_to_right, steps_to_bottom))
    
    # Determine sampling parameters
    max_overall_steps = np.max(max_steps_per_angle)
    if max_overall_steps <= 0:
        return np.full(n_angles, float(array[int(start_row), int(start_col)]))
    
    num_steps = int(max_overall_steps / step_size) + 1
    t_values = np.linspace(0, max_overall_steps, num_steps)  # Shape: (num_steps,)
    
    # Generate all coordinates: (num_steps, n_angles)
    all_rows = start_row + t_values[:, None] * dy[None, :]
    all_cols = start_col + t_values[:, None] * dx[None, :]
    
    # Convert to integer coordinates
    row_indices = np.round(all_rows).astype(int)
    col_indices = np.round(all_cols).astype(int)
    
    # Create validity masks
    within_bounds = (
        (row_indices >= 0) & (row_indices < rows) &
        (col_indices >= 0) & (col_indices < cols)
    )
    within_ray_length = t_values[:, None] <= max_steps_per_angle[None, :]
    valid_mask = within_bounds & within_ray_length
    
    # Vectorized duplicate removal using coordinate differences
    # Find where coordinates change from one step to the next
    row_diff = np.diff(row_indices, axis=0, prepend=row_indices[0:1] - 1)
    col_diff = np.diff(col_indices, axis=0, prepend=col_indices[0:1] - 1)
    coord_changed = (row_diff != 0) | (col_diff != 0)
    
    # Use coordinates only where they're valid AND have changed
    use_coord = valid_mask & coord_changed
    
    # Clamp coordinates to valid range to avoid indexing errors
    safe_rows = np.clip(row_indices, 0, rows - 1)
    safe_cols = np.clip(col_indices, 0, cols - 1)
    
    # Get pixel values (invalid/unused coords contribute 0)
    pixel_values = np.where(use_coord, array[safe_rows, safe_cols], 0)
    
    # Sum along rays (axis 0 = steps)
    return np.sum(pixel_values, axis=0)

class Eye:

    def __init__(self, board, verg_angle, retinal_field, is_left, env_variables, max_uv_range,
                 plot_rfs=False):
        # Use CUPY if using GPU.
        self.test_mode = env_variables["test_sensory_system"]
        
        self.viewing_elevations = env_variables["viewing_elevations"]
        self.fish_elevation = env_variables["elevation"]
        self.uv_object_intensity = env_variables["uv_object_intensity"]
        self.water_uv_scatter = env_variables["water_uv_scatter"]
        self.board = board
        # self.dark_gain = env_variables['dark_gain']
        # self.light_gain = env_variables['light_gain']
        # self.background_brightness = env_variables['background_brightness']
        #self.dark_col = dark_col
        self.dist = None
        self.theta = None
        self.width, self.height = self.board.get_FOV_size()
        self.retinal_field_size = retinal_field
        self.env_variables = env_variables
        self.max_uv_range = max_uv_range
        self.prey_diameter = self.env_variables['prey_radius'] * 2

        self.sz_rf_spacing = self.env_variables["sz_rf_spacing"]
        self.sz_size = self.env_variables["sz_size"]
        self.sz_oversampling_factor = self.env_variables["sz_oversampling_factor"]
        self.sigmoid_steepness = self.env_variables["sigmoid_steepness"]

        self.periphery_rf_spacing = self.sz_rf_spacing * self.sz_oversampling_factor
        self.density_range = self.periphery_rf_spacing - self.sz_rf_spacing

        self.ang_bin = 0.001  # this is the bin size for the projection
        self.ang = np.arange(-np.pi, np.pi + self.ang_bin,
                                                   self.ang_bin)  # this is the angle range for the projection

        self.filter_bins = 20  # this is the number of bins for the scattering filter

        self.uv_photoreceptor_rf_size = env_variables['uv_photoreceptor_rf_size']
        self.red_photoreceptor_rf_size = env_variables['red_photoreceptor_rf_size']

        self.uv_photoreceptor_angles = self.update_angles_sigmoid(verg_angle, retinal_field, is_left)
        self.uv_photoreceptor_num = len(self.uv_photoreceptor_angles)
        self.red_photoreceptor_num = self.uv_photoreceptor_num

        self.interpolated_observation = np.arange(
            np.min(self.uv_photoreceptor_angles),
            np.max(self.uv_photoreceptor_angles) + self.sz_rf_spacing / 2,
            self.sz_rf_spacing / 2)

        self.observation_size = len(self.interpolated_observation)

        self.red_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left,
                                                           self.red_photoreceptor_num)

        self.uv_readings = np.zeros((self.uv_photoreceptor_num, 1))
        self.red_readings = np.zeros((self.red_photoreceptor_num, 1))
        self.total_photoreceptor_num = self.uv_photoreceptor_num + self.red_photoreceptor_num
        # Compute minimum lines that need to be extrapolated so as not to miss coordinates.
        #self.n = self.compute_n(max([self.uv_photoreceptor_rf_size, self.red_photoreceptor_rf_size]))

        if plot_rfs:
            self.plot_photoreceptors(self.uv_photoreceptor_angles.get(), self.red_photoreceptor_angles.get(),
                                        self.uv_photoreceptor_rf_size, self.red_photoreceptor_rf_size, is_left)
        # Compute repeated measures:
        self.photoreceptor_angles_surrounding = None
        self.photoreceptor_angles_surrounding_red = None
        self.mul_for_hypothetical = None
        self.add_for_hypothetical = None
        self.mul1_full = None
        self.addition_matrix = None
        self.conditional_tiled = None
        self.multiplication_matrix = None
        #self.get_repeated_computations()

        # self.photoreceptor_angles_surrounding_stacked = np.concatenate((self.photoreceptor_angles_surrounding,
        #                                                                                       self.photoreceptor_angles_surrounding_red),
        #                                                                                      axis=0)

    @staticmethod
    def plot_photoreceptors(uv_photoreceptor_angles, red_photoreceptor_angles, uv_photoreceptor_rf_size,
                            red_photoreceptor_rf_size, is_left):
        # Plot the photoreceptors on a polar plot:
        plt.ioff()

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_yticklabels([])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        uv_rfs = np.zeros((len(uv_photoreceptor_angles), 2))
        red_rfs = np.zeros((len(red_photoreceptor_angles), 2))
        uv_rfs[:, 0] = uv_photoreceptor_angles - uv_photoreceptor_rf_size / 2
        uv_rfs[:, 1] = uv_photoreceptor_angles + uv_photoreceptor_rf_size / 2
        red_rfs[:, 0] = red_photoreceptor_angles - red_photoreceptor_rf_size / 2
        red_rfs[:, 1] = red_photoreceptor_angles + red_photoreceptor_rf_size / 2
        r_uv = np.ones(uv_rfs.shape) * 0.9
        r_red = np.ones(red_rfs.shape) * 1.1
        ax.plot(uv_rfs.T, r_uv.T, color='b', alpha=0.3, linewidth=1)
        ax.plot(red_rfs.T, r_red.T, color='r', alpha=0.3, linewidth=1)
        if is_left:
            plt.savefig('left_eye.png')
        else:
            plt.savefig('right_eye.png')
        fig.clf()
        # plt.ion()

    def get_repeated_computations(self):
        """
        Pre-computes and stores in memory all sets of repeated values for the read_stacked method:
           - self.photoreceptor_angles_surrounding - The angles, from the midline of the fish, which enclose the active
           region of each UV photoreceptor
           - self.photoreceptor_angles_surrounding_red  - " but for all red photoreceptors
           - self.mul_for_hypothetical
           - self.add_for_hypothetical
           - self.mul1_full
           - self.addition_matrix
           - self.conditional_tiled
           - self.multiplication_matrix

        """

        # UV
        photoreceptor_angles_surrounding = np.expand_dims(self.uv_photoreceptor_angles, 1)
        photoreceptor_angles_surrounding = np.repeat(photoreceptor_angles_surrounding, self.n, 1)
        rf_offsets = np.linspace(-self.uv_photoreceptor_rf_size / 2,
                                                       self.uv_photoreceptor_rf_size / 2, num=self.n)
        self.photoreceptor_angles_surrounding = photoreceptor_angles_surrounding + rf_offsets

        # Red
        photoreceptor_angles_surrounding_2 = np.expand_dims(self.red_photoreceptor_angles, 1)
        photoreceptor_angles_surrounding_2 = np.repeat(photoreceptor_angles_surrounding_2, self.n, 1)
        rf_offsets_2 = np.linspace(-self.red_photoreceptor_rf_size / 2,
                                                         self.red_photoreceptor_rf_size / 2, num=self.n)
        self.photoreceptor_angles_surrounding_red = photoreceptor_angles_surrounding_2 + rf_offsets_2

        n_photoreceptors_in_computation_axis_0 = self.total_photoreceptor_num

        # Same for both, just requires different dimensions
        mul_for_hypothetical = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = np.tile(mul_for_hypothetical,
                                                                  (n_photoreceptors_in_computation_axis_0, self.n, 1, 1))
        add_for_hypothetical = np.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        self.add_for_hypothetical = np.tile(add_for_hypothetical,
                                                                  (n_photoreceptors_in_computation_axis_0, self.n, 1, 1))

        mul1 = np.array([0, 0, 0, 1])
        self.mul1_full = np.tile(mul1, (n_photoreceptors_in_computation_axis_0, self.n, 1))

        addition_matrix_unit = np.array([0, 0, self.height - 1, self.width - 1])
        self.addition_matrix = np.tile(addition_matrix_unit,
                                                             (n_photoreceptors_in_computation_axis_0, self.n, 1))

        conditional_tiled = np.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        self.conditional_tiled = np.tile(conditional_tiled,
                                                               (n_photoreceptors_in_computation_axis_0, self.n, 1))

        multiplication_matrix_unit = np.array([-1, 1, -1, 1])
        self.multiplication_matrix = np.tile(multiplication_matrix_unit,
                                                                   (n_photoreceptors_in_computation_axis_0, self.n, 1))

    def update_angles_sigmoid(self, verg_angle, retinal_field, is_left):
        """Set the eyes visual angles to be a sigmoidal distribution."""

        pr = [0]
        while True:
            spacing = self.sz_rf_spacing + self.density_range / (
                    1 + np.exp(-self.sigmoid_steepness * (pr[-1] - self.sz_size)))
            pr.append(pr[-1] + spacing)
            if pr[-1] > retinal_field:
                break
        pr = pr[:-1]
        pr = np.array(pr)
        if is_left:
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
            pr = max_angle - pr
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            pr = pr + min_angle

        return np.sort(pr)

    def update_angles(self, verg_angle, retinal_field, is_left, photoreceptor_num):
        """Set the eyes visual angles to be an even distribution."""
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        return np.linspace(min_angle, max_angle, photoreceptor_num)

    def read(self, masked_arena_pixels, eye_x, eye_y, fish_angle, uv_lum_mask, prey_positions, sand_grain_positions, predator_left, predator_right, predator_dist,
             proj=True):
        """
        Resolve RF coordinates for each photoreceptor, and use those to sum the relevant pixels.
        """
        # UV Angles with respect to fish (doubled) (PR_N x n)

        # photoreceptor_angles_surrounding = self.photoreceptor_angles_surrounding_stacked + fish_angle
        # uv_arena_pixels = masked_arena_pixels[:, :, 1:2]
        # red_arena_pixels = np.concatenate(
        #     (masked_arena_pixels[:, :, 0:1], masked_arena_pixels[:, :, 2:]), axis=2)
        # uv_readings, red_readings = self._read_stacked(masked_arena_pixels_uv=uv_arena_pixels,
        #                                                masked_arena_pixels_red=red_arena_pixels,
        #                                                eye_x=eye_x,
        #                                                eye_y=eye_y,
        #                                                photoreceptor_angles_surrounding=photoreceptor_angles_surrounding,
        #                                                n_photoreceptors_uv=self.uv_photoreceptor_num,
        #                                                n_photoreceptors_red=self.red_photoreceptor_num)

        #uv_readings = np.zeros((self.uv_photoreceptor_num, 1))
        corrected_uv_pr_angles = np.arctan2(np.sin(self.uv_photoreceptor_angles + fish_angle),
                                                                np.cos(self.uv_photoreceptor_angles + fish_angle))
        eye_FOV_x = eye_x + (uv_lum_mask.shape[1] - 1) / 2
        eye_FOV_y = eye_y + (uv_lum_mask.shape[0] - 1) / 2

        self.uv_readings = self.water_uv_scatter * np.expand_dims(ray_sum_fully_vectorized(uv_lum_mask, (eye_FOV_y, eye_FOV_x), corrected_uv_pr_angles), axis=1)
        self.red_readings = np.zeros((self.red_photoreceptor_num, 2))
        for ii, view_elevation in enumerate(self.viewing_elevations):
            self.red_readings[:, ii] = self._read_elevation(masked_arena_pixels, eye_x, eye_y, view_elevation, fish_angle,
                                 self.red_photoreceptor_angles, self.red_photoreceptor_rf_size, predator_left, predator_right, predator_dist)

        if len(sand_grain_positions) > 0:
            uv_items = np.concatenate((prey_positions, sand_grain_positions), axis=0)
        else:
            uv_items = prey_positions
        if proj and (len(uv_items)) > 0:
            proj_uv_readings = self._read_prey_proj(eye_x=eye_x,
                                                    eye_y=eye_y,
                                                    uv_pr_angles=self.uv_photoreceptor_angles,
                                                    fish_angle=fish_angle,
                                                    rf_size=self.uv_photoreceptor_rf_size,
                                                    lum_mask=uv_lum_mask,
                                                    prey_pos=np.array(uv_items))
            proj_uv_readings *= self.uv_object_intensity

            self.uv_readings += proj_uv_readings
            if len(sand_grain_positions) > 0:
                red_readings_sand_grains = self._read_prey_proj(eye_x=eye_x,
                                                                eye_y=eye_y,
                                                                uv_pr_angles=self.red_photoreceptor_angles,
                                                                fish_angle=fish_angle,
                                                                rf_size=self.red_photoreceptor_rf_size,
                                                                lum_mask=lum_mask,
                                                                prey_pos=np.array(sand_grain_positions)
                                                                )

                self.red_readings += red_readings_sand_grains * self.env_variables["sand_grain_red_component"]

        #if not self.test_mode:
        self.uv_readings = self.add_noise_to_readings(self.uv_readings)
        self.red_readings = self.add_noise_to_readings(self.red_readings)

        interp_uv_readings = np.zeros((self.interpolated_observation.shape[0], 1))
        interp_red_readings = np.zeros((self.interpolated_observation.shape[0], 2))
        interp_uv_readings[:, 0] = np.interp(self.interpolated_observation,
                                                                   self.uv_photoreceptor_angles, self.uv_readings[:, 0])
        interp_red_readings[:, 0] = np.interp(self.interpolated_observation,
                                                                    self.red_photoreceptor_angles,
                                                                    self.red_readings[:, 0])
        interp_red_readings[:, 1] = np.interp(self.interpolated_observation,
                                                                    self.red_photoreceptor_angles,
                                                                    self.red_readings[:, 1])

        self.readings = np.concatenate(
            (interp_uv_readings, interp_red_readings), axis=1)
        
        # if np == cp:
        #     self.readings = self.readings.get()

    def _read_elevation(self, masked_pixels, eye_x, eye_y, elevation_angle, fish_angle, pr_angles, rf_size, predator_left, predator_right, predator_dist):


        eye_FOV_x = int(eye_x + (masked_pixels.shape[1] - 1) / 2)
        eye_FOV_y = int(eye_y + (masked_pixels.shape[0] - 1) / 2)
        radius = int(np.round(self.fish_elevation * np.tan(np.radians(elevation_angle))))
        circle_values, circle_angles = extract_circle_edge_pixels(masked_pixels, (eye_FOV_y, eye_FOV_x), radius)
        if ~np.isnan(predator_dist):
            if predator_dist < radius:
                if predator_left > predator_right:
                    circle_values[(circle_angles < predator_left) & (circle_angles > predator_right)] = 0
                else:
                    circle_values[(circle_angles < predator_left) | (circle_angles > predator_right)] = 0
        corrected_pr_angles = np.arctan2(np.sin(pr_angles + fish_angle),
                                                                np.cos(pr_angles + fish_angle))
        bin_edges1 = corrected_pr_angles - rf_size / 2
        bin_edges2 = corrected_pr_angles + rf_size / 2


        # find closest circle angle to each bin edge
        l_ind = self._closest_index_parallel(circle_angles, bin_edges1)
        r_ind = self._closest_index_parallel(circle_angles, bin_edges2) + 1

        interleaved_edges = np.vstack((l_ind, r_ind)).T.flatten()
        bins_sum = np.add.reduceat(circle_values, interleaved_edges)[::2]
        counts = r_ind - l_ind
        pr_input = bins_sum / counts
        return pr_input
        
        
        

    def _read_prey_proj(self, eye_x, eye_y, uv_pr_angles, fish_angle, rf_size, lum_mask, prey_pos):
        """Reads the prey projection for the given eye position and fish angle.
        Same as " but performs more computation in parallel for each prey. Also have removed scatter.
        """
        eye_FOV_x = eye_x + (lum_mask.shape[1] - 1) / 2
        eye_FOV_y = eye_y + (lum_mask.shape[0] - 1) / 2
        rel_prey_pos = prey_pos - np.array([eye_FOV_x, eye_FOV_y])
        rho = np.hypot(rel_prey_pos[:, 0], rel_prey_pos[:, 1])

        within_range = np.where(rho < self.max_uv_range - 1)[0]
        prey_pos_in_range = prey_pos[within_range, :]
        rel_prey_pos = rel_prey_pos[within_range, :]
        rho = rho[within_range]
        theta = np.arctan2(rel_prey_pos[:, 1], rel_prey_pos[:, 0]) - fish_angle
        theta = np.arctan2(np.sin(theta),
                                                 np.cos(theta))  # wrap to [-pi, pi]
        p_num = prey_pos_in_range.shape[0]

        half_angle = np.arctan(self.prey_diameter / (2 * rho))

        l_ind = self._closest_index_parallel(self.ang, theta - half_angle).astype(int)
        r_ind = self._closest_index_parallel(self.ang, theta + half_angle).astype(int)

        prey_brightness = lum_mask[(np.floor(prey_pos_in_range[:, 1]) - 1).astype(int),
                                   (np.floor(prey_pos_in_range[:, 0]) - 1).astype(
                                       int)]  # includes absorption (???)

        proj = np.zeros((p_num, len(self.ang)))

        prey_brightness = np.expand_dims(prey_brightness, 1)

        r = np.arange(proj.shape[1])
        prey_present = (l_ind[:, None] <= r) & (r_ind[:, None] >= r)
        prey_present = prey_present.astype(float)
        prey_present *= prey_brightness

        total_angular_input = np.sum(prey_present, axis=0)

        pr_ind_s = self._closest_index_parallel(self.ang, uv_pr_angles - rf_size / 2)
        pr_ind_e = self._closest_index_parallel(self.ang, uv_pr_angles + rf_size / 2)

        pr_occupation = (pr_ind_s[:, None] <= r) & (pr_ind_e[:, None] >= r)
        pr_occupation = pr_occupation.astype(float)
        pr_input = pr_occupation * np.expand_dims(total_angular_input, axis=0)
        pr_input = np.sum(pr_input, axis=1)

        return np.expand_dims(pr_input, axis=1)

    def _closest_index_parallel(self, array, value_array):
        """Find indices of the closest values in array (for each row in axis=0)."""
        value_array = np.expand_dims(value_array, axis=1)
        idxs = (np.abs(array - value_array)).argmin(axis=1)
        return idxs


    def add_noise_to_readings(self, readings):
        """Samples from Poisson distribution to get number of photons"""
        if self.env_variables["shot_noise"]:
            photons = np.random.poisson(readings)
        else:
            photons = readings

        return photons

    def compute_n(self, photoreceptor_rf_size, max_separation=1):
        theta_separation = math.asin(max_separation / self.max_visual_range)
        n = (photoreceptor_rf_size / theta_separation)
        return int(n)



def extract_circle_edge_pixels(image: np.ndarray, center: tuple, radius: int):
    """
    Extract pixel values from the edge of a circular region in an image.
    
    Parameters:
        image (np.ndarray): Input image as a NumPy array.
        center (tuple): (row, col) coordinates of the circle's center.
        radius (int): Radius of the circular edge.
    
    Returns:
        tuple: (Extracted pixel values, Corresponding angles in radians)
    """
    rr, cc = circle_perimeter(center[0], center[1], radius, shape=image.shape[:2])
    pixels = image[rr, cc]
    
    angles = np.arctan2(rr - center[0], cc - center[1])
    
    pixels = pixels[angles.argsort()]
    angles = np.sort(angles)
    return pixels, angles
