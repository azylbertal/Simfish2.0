import numpy as np
import math
import matplotlib.pyplot as plt
import cupy as cp


class Eye:

    def __init__(self, board, verg_angle, retinal_field, is_left, env_variables, dark_col, max_visual_range,
                 plot_rfs=False):
        # Use CUPY if using GPU.
        self.chosen_math_library = np

        self.uv_object_intensity = env_variables["uv_object_intensity"]
        self.board = board
        self.dark_gain = env_variables['dark_gain']
        self.light_gain = env_variables['light_gain']
        self.background_brightness = env_variables['background_brightness']
        self.dark_col = dark_col
        self.dist = None
        self.theta = None
        self.width, self.height = self.board.get_FOV_size()
        self.retinal_field_size = retinal_field
        self.env_variables = env_variables
        self.max_visual_range = max_visual_range
        self.prey_diameter = self.env_variables['prey_radius'] * 2

        self.sz_rf_spacing = self.env_variables["sz_rf_spacing"]
        self.sz_size = self.env_variables["sz_size"]
        self.sz_oversampling_factor = self.env_variables["sz_oversampling_factor"]
        self.sigmoid_steepness = self.env_variables["sigmoid_steepness"]

        self.periphery_rf_spacing = self.sz_rf_spacing * self.sz_oversampling_factor
        self.density_range = self.periphery_rf_spacing - self.sz_rf_spacing

        self.ang_bin = 0.001  # this is the bin size for the projection
        self.ang = self.chosen_math_library.arange(-np.pi, np.pi + self.ang_bin,
                                                   self.ang_bin)  # this is the angle range for the projection

        self.filter_bins = 20  # this is the number of bins for the scattering filter

        self.uv_photoreceptor_rf_size = env_variables['uv_photoreceptor_rf_size']
        self.red_photoreceptor_rf_size = env_variables['red_photoreceptor_rf_size']

        self.uv_photoreceptor_angles = self.update_angles_sigmoid(verg_angle, retinal_field, is_left)
        self.uv_photoreceptor_num = len(self.uv_photoreceptor_angles)
        self.red_photoreceptor_num = self.uv_photoreceptor_num

        self.interpolated_observation = self.chosen_math_library.arange(
            self.chosen_math_library.min(self.uv_photoreceptor_angles),
            self.chosen_math_library.max(self.uv_photoreceptor_angles) + self.sz_rf_spacing / 2,
            self.sz_rf_spacing / 2)

        self.observation_size = len(self.interpolated_observation)

        self.red_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left,
                                                           self.red_photoreceptor_num)

        self.uv_readings = self.chosen_math_library.zeros((self.uv_photoreceptor_num, 1))
        self.red_readings = self.chosen_math_library.zeros((self.red_photoreceptor_num, 1))
        self.total_photoreceptor_num = self.uv_photoreceptor_num + self.red_photoreceptor_num
        # Compute minimum lines that need to be extrapolated so as not to miss coordinates.
        self.n = self.compute_n(max([self.uv_photoreceptor_rf_size, self.red_photoreceptor_rf_size]))

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
        self.get_repeated_computations()

        self.photoreceptor_angles_surrounding_stacked = self.chosen_math_library.concatenate((self.photoreceptor_angles_surrounding,
                                                                                              self.photoreceptor_angles_surrounding_red),
                                                                                             axis=0)

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
        photoreceptor_angles_surrounding = self.chosen_math_library.expand_dims(self.uv_photoreceptor_angles, 1)
        photoreceptor_angles_surrounding = self.chosen_math_library.repeat(photoreceptor_angles_surrounding, self.n, 1)
        rf_offsets = self.chosen_math_library.linspace(-self.uv_photoreceptor_rf_size / 2,
                                                       self.uv_photoreceptor_rf_size / 2, num=self.n)
        self.photoreceptor_angles_surrounding = photoreceptor_angles_surrounding + rf_offsets

        # Red
        photoreceptor_angles_surrounding_2 = self.chosen_math_library.expand_dims(self.red_photoreceptor_angles, 1)
        photoreceptor_angles_surrounding_2 = self.chosen_math_library.repeat(photoreceptor_angles_surrounding_2, self.n, 1)
        rf_offsets_2 = self.chosen_math_library.linspace(-self.red_photoreceptor_rf_size / 2,
                                                         self.red_photoreceptor_rf_size / 2, num=self.n)
        self.photoreceptor_angles_surrounding_red = photoreceptor_angles_surrounding_2 + rf_offsets_2

        n_photoreceptors_in_computation_axis_0 = self.total_photoreceptor_num

        # Same for both, just requires different dimensions
        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical,
                                                                  (n_photoreceptors_in_computation_axis_0, self.n, 1, 1))
        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        self.add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical,
                                                                  (n_photoreceptors_in_computation_axis_0, self.n, 1, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        self.mul1_full = self.chosen_math_library.tile(mul1, (n_photoreceptors_in_computation_axis_0, self.n, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.height - 1, self.width - 1])
        self.addition_matrix = self.chosen_math_library.tile(addition_matrix_unit,
                                                             (n_photoreceptors_in_computation_axis_0, self.n, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        self.conditional_tiled = self.chosen_math_library.tile(conditional_tiled,
                                                               (n_photoreceptors_in_computation_axis_0, self.n, 1))

        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        self.multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit,
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
        pr = self.chosen_math_library.array(pr)
        if is_left:
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
            pr = max_angle - pr
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            pr = pr + min_angle

        return self.chosen_math_library.sort(pr)

    def update_angles(self, verg_angle, retinal_field, is_left, photoreceptor_num):
        """Set the eyes visual angles to be an even distribution."""
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        return self.chosen_math_library.linspace(min_angle, max_angle, photoreceptor_num)

    def read(self, masked_arena_pixels, eye_x, eye_y, fish_angle, lum_mask, prey_positions, sand_grain_positions,
             proj=True):
        """
        Resolve RF coordinates for each photoreceptor, and use those to sum the relevant pixels.
        """
        # UV Angles with respect to fish (doubled) (PR_N x n)

        # photoreceptor_angles_surrounding = self.photoreceptor_angles_surrounding_stacked + fish_angle
        # uv_arena_pixels = masked_arena_pixels[:, :, 1:2]
        # red_arena_pixels = self.chosen_math_library.concatenate(
        #     (masked_arena_pixels[:, :, 0:1], masked_arena_pixels[:, :, 2:]), axis=2)
        # uv_readings, red_readings = self._read_stacked(masked_arena_pixels_uv=uv_arena_pixels,
        #                                                masked_arena_pixels_red=red_arena_pixels,
        #                                                eye_x=eye_x,
        #                                                eye_y=eye_y,
        #                                                photoreceptor_angles_surrounding=photoreceptor_angles_surrounding,
        #                                                n_photoreceptors_uv=self.uv_photoreceptor_num,
        #                                                n_photoreceptors_red=self.red_photoreceptor_num)

        uv_readings = self.chosen_math_library.zeros((self.uv_photoreceptor_num, 1))
        red_readings = self.chosen_math_library.zeros((self.red_photoreceptor_num, 2))
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
                                                    lum_mask=lum_mask,
                                                    prey_pos=self.chosen_math_library.array(uv_items))
            proj_uv_readings *= self.uv_object_intensity

            uv_readings += proj_uv_readings
            if len(sand_grain_positions) > 0:
                red_readings_sand_grains = self._read_prey_proj(eye_x=eye_x,
                                                                eye_y=eye_y,
                                                                uv_pr_angles=self.red_photoreceptor_angles,
                                                                fish_angle=fish_angle,
                                                                rf_size=self.red_photoreceptor_rf_size,
                                                                lum_mask=lum_mask,
                                                                prey_pos=self.chosen_math_library.array(sand_grain_positions)
                                                                )

                red_readings += red_readings_sand_grains * self.env_variables["sand_grain_red_component"]

        self.uv_readings = self.add_noise_to_readings(uv_readings)
        self.red_readings = self.add_noise_to_readings(red_readings)

        interp_uv_readings = self.chosen_math_library.zeros((self.interpolated_observation.shape[0], 1))
        interp_red_readings = self.chosen_math_library.zeros((self.interpolated_observation.shape[0], 2))
        interp_uv_readings[:, 0] = self.chosen_math_library.interp(self.interpolated_observation,
                                                                   self.uv_photoreceptor_angles, self.uv_readings[:, 0])
        interp_red_readings[:, 0] = self.chosen_math_library.interp(self.interpolated_observation,
                                                                    self.red_photoreceptor_angles,
                                                                    self.red_readings[:, 0])
        interp_red_readings[:, 1] = self.chosen_math_library.interp(self.interpolated_observation,
                                                                    self.red_photoreceptor_angles,
                                                                    self.red_readings[:, 1])

        self.readings = self.chosen_math_library.concatenate(
            (interp_red_readings[:, 0:1], interp_uv_readings, interp_red_readings[:, 1:]), axis=1)
        
        if self.chosen_math_library == cp:
            self.readings = self.readings.get()

    def _read_prey_proj(self, eye_x, eye_y, uv_pr_angles, fish_angle, rf_size, lum_mask, prey_pos):
        """Reads the prey projection for the given eye position and fish angle.
        Same as " but performs more computation in parallel for each prey. Also have removed scatter.
        """

        rel_prey_pos = prey_pos - self.chosen_math_library.array([eye_x, eye_y])
        rho = self.chosen_math_library.hypot(rel_prey_pos[:, 0], rel_prey_pos[:, 1])

        within_range = self.chosen_math_library.where(rho < self.max_visual_range - 1)[0]
        prey_pos_in_range = prey_pos[within_range, :]
        rel_prey_pos = rel_prey_pos[within_range, :]
        rho = rho[within_range]
        theta = self.chosen_math_library.arctan2(rel_prey_pos[:, 1], rel_prey_pos[:, 0]) - fish_angle
        theta = self.chosen_math_library.arctan2(self.chosen_math_library.sin(theta),
                                                 self.chosen_math_library.cos(theta))  # wrap to [-pi, pi]
        p_num = prey_pos_in_range.shape[0]

        half_angle = self.chosen_math_library.arctan(self.prey_diameter / (2 * rho))

        l_ind = self._closest_index_parallel(self.ang, theta - half_angle).astype(int)
        r_ind = self._closest_index_parallel(self.ang, theta + half_angle).astype(int)

        prey_brightness = lum_mask[(self.chosen_math_library.floor(prey_pos_in_range[:, 1]) - 1).astype(int),
                                   (self.chosen_math_library.floor(prey_pos_in_range[:, 0]) - 1).astype(
                                       int)]  # includes absorption

        proj = self.chosen_math_library.zeros((p_num, len(self.ang)))

        prey_brightness = self.chosen_math_library.expand_dims(prey_brightness, 1)

        r = self.chosen_math_library.arange(proj.shape[1])
        prey_present = (l_ind[:, None] <= r) & (r_ind[:, None] >= r)
        prey_present = prey_present.astype(float)
        prey_present *= prey_brightness

        total_angular_input = self.chosen_math_library.sum(prey_present, axis=0)

        pr_ind_s = self._closest_index_parallel(self.ang, uv_pr_angles - rf_size / 2)
        pr_ind_e = self._closest_index_parallel(self.ang, uv_pr_angles + rf_size / 2)

        pr_occupation = (pr_ind_s[:, None] <= r) & (pr_ind_e[:, None] >= r)
        pr_occupation = pr_occupation.astype(float)
        pr_input = pr_occupation * self.chosen_math_library.expand_dims(total_angular_input, axis=0)
        pr_input = self.chosen_math_library.sum(pr_input, axis=1)

        return self.chosen_math_library.expand_dims(pr_input, axis=1)

    def _closest_index_parallel(self, array, value_array):
        """Find indices of the closest values in array (for each row in axis=0)."""
        value_array = self.chosen_math_library.expand_dims(value_array, axis=1)
        idxs = (self.chosen_math_library.abs(array - value_array)).argmin(axis=1)
        return idxs

    def _read_stacked(self, masked_arena_pixels_uv, masked_arena_pixels_red, eye_x, eye_y, photoreceptor_angles_surrounding,
                      n_photoreceptors_uv, n_photoreceptors_red):
        """
        Lines method to return pixel sum for all points for each photoreceptor, over its segment.
        Takes photoreceptor_angles_surrounding with UV then Red.

        The aim of this method is to find all the coordinates enclosed in the receptive field of each photoreceptor.
        This is done by defining n lines per photoreceptor, which are linearly interpolated between the edges of each photoreceptor,
        tracing them for as far as they extend, and so find all valid coordinates which fall on them.
        This necessarily over samples (disproportionately for shorter lines), though predictably, so can be accounted
        for.

        The method is poorly organised, as it has been designed to have as little computational overhead as possible.
        To be efficient, the method must perform its computations in parallel (which limits the way of doing it to
        memory inefficient and verbose approaches).
        """
        n_photoreceptors = n_photoreceptors_uv + n_photoreceptors_red

        # Make sure angles are in desired range (PR_N x n)
        photoreceptor_angles_surrounding_scaling = (photoreceptor_angles_surrounding // (
                self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        photoreceptor_angles_surrounding = photoreceptor_angles_surrounding + photoreceptor_angles_surrounding_scaling

        # Compute m (gradient of lines enclosing each PR) using tan (PR_N x n)
        m = self.chosen_math_library.tan(photoreceptor_angles_surrounding)

        # Compute c (y-intercept of lines enclosing each PR) (PR_N x n)
        c = -m * eye_x
        c = c + eye_y

        # Compute components of intersections (two for each axis - up and down) (PR_N x n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 2)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 2)

        m_mul = self.chosen_math_library.expand_dims(m, 2)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 2)
        m_mul = full_m * self.mul1_full[:n_photoreceptors]
        m_mul[:, :, :3] = 1
        addition_matrix = self.addition_matrix[:n_photoreceptors] * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c_exp * self.multiplication_matrix[:n_photoreceptors]) + addition_matrix) / division_matrix

        # Compute the coordinates at which the lines intersect the bounds of the environment/visual range. Note that
        # some of these may be outside the environment along one axis
        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[:n_photoreceptors]) +\
                                   self.add_for_hypothetical[:n_photoreceptors]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:n_photoreceptors]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates,
                                                                          (n_photoreceptors, self.n, 2, 2))

        # Compute the possible emanating vectors of the enclosing line of each photoreceptor (PR_N x 2)
        eye_position = self.chosen_math_library.array([eye_x, eye_y])
        possible_vectors = valid_intersection_coordinates - eye_position

        # Compute the angles of each possible vector
        angles = self.chosen_math_library.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range.
        angle_scaling = (angles // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        angles = angles + angle_scaling

        # Round all angles to allow direct comparison.
        angles = self.chosen_math_library.round(angles, 3)
        photoreceptor_angles_surrounding = self.chosen_math_library.round(photoreceptor_angles_surrounding, 3)
        photoreceptor_angles_surrounding = self.chosen_math_library.expand_dims(photoreceptor_angles_surrounding, 2)
        photoreceptor_angles_surrounding = self.chosen_math_library.repeat(photoreceptor_angles_surrounding, 2, 2)

        # Find which of the hypothetical angles match the original specified angles.
        same_values = (angles == photoreceptor_angles_surrounding) * 1

        # Thus, outline the actual intersections with the bounding environment/visual range
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = self.chosen_math_library.reshape(selected_intersections,
                                                                  (n_photoreceptors, self.n, 1, 2))

        eye_position_full = self.chosen_math_library.tile(eye_position, (n_photoreceptors, self.n, 1, 1))
        vertices = self.chosen_math_library.concatenate((eye_position_full, selected_intersections), axis=2)
        vertices_xvals = vertices[:, :, :, 0]
        vertices_yvals = vertices[:, :, :, 1]

        min_x = self.chosen_math_library.min(vertices_xvals, axis=2)
        max_x = self.chosen_math_library.max(vertices_xvals, axis=2)
        min_y = self.chosen_math_library.min(vertices_yvals, axis=2)
        max_y = self.chosen_math_library.max(vertices_yvals, axis=2)

        # Threshold with distance from fish
        absolute_min_x = eye_x - self.max_visual_range
        absolute_max_x = eye_x + self.max_visual_range
        absolute_min_y = eye_y - self.max_visual_range
        absolute_max_y = eye_y + self.max_visual_range
        min_x = self.chosen_math_library.clip(min_x, absolute_min_x, self.width)
        max_x = self.chosen_math_library.clip(max_x, 0, absolute_max_x)
        min_y = self.chosen_math_library.clip(min_y, absolute_min_y, self.height)
        max_y = self.chosen_math_library.clip(max_y, 0, absolute_max_y)

        # SEGMENT COMPUTATION
        x_lens = self.chosen_math_library.rint(max_x[:, 0] - min_x[:, 0])
        y_lens = self.chosen_math_library.rint(max_y[:, 0] - min_y[:, 0])

        x_len = self.chosen_math_library.max(x_lens)
        y_len = self.chosen_math_library.max(y_lens)

        # Find appropriate coordinates between the outlined intersections along which to evaluate the lines
        x_ranges = self.chosen_math_library.linspace(min_x, max_x, int(x_len))
        y_ranges = self.chosen_math_library.linspace(min_y, max_y, int(y_len))

        # Evaluate the lines along the ranges, producing the final sets of coordinates enclosed by the photoreceptor RFs.
        y_values = (m * x_ranges) + c
        y_values = self.chosen_math_library.floor(y_values)
        set_1 = self.chosen_math_library.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = self.chosen_math_library.floor(x_values)
        set_2 = self.chosen_math_library.stack((x_values, y_ranges), axis=-1)
        full_set = self.chosen_math_library.vstack((set_1, set_2)).astype(int)

        full_set = full_set.swapaxes(0, 1)
        full_set = full_set.reshape(n_photoreceptors, -1, 2)

        full_set_uv = full_set[:n_photoreceptors_uv, :]
        full_set_red = full_set[n_photoreceptors_uv:, :]

        # For each set of coordinates, get the pixels in that region of the environment.
        masked_arena_pixels_uv = masked_arena_pixels_uv[full_set_uv[:, :, 1],
                                                        full_set_uv[:, :, 0]]  # NOTE: Inverts x and y to match standard.
        total_sum_uv = masked_arena_pixels_uv.sum(axis=1)

        masked_arena_pixels_red = masked_arena_pixels_red[full_set_red[:, :, 1],
                                                          full_set_red[:, :, 0]]  # NOTE: Inverts x and y to match standard.
        total_sum_red = masked_arena_pixels_red.sum(axis=1)

        # Compute oversampling ratio. This takes account of how many indexes have been computed for each sector, and
        # scales all by this so there is an even density of pixel counts (otherwise small rays would be counted more).
        # (PR_N)
        oversampling_ratio = (x_lens + y_lens) / (x_len + y_len)
        oversampling_ratio = self.chosen_math_library.expand_dims(oversampling_ratio, 1)
        oversampling_ratio_red = self.chosen_math_library.repeat(oversampling_ratio, total_sum_red.shape[1], 1)

        total_sum_uv = total_sum_uv * (oversampling_ratio[:n_photoreceptors_uv] / self.n)
        total_sum_red = total_sum_red * (oversampling_ratio_red[n_photoreceptors_uv:] / self.n)

        return total_sum_uv, total_sum_red

    def add_noise_to_readings(self, readings):
        """Samples from Poisson distribution to get number of photons"""
        if self.env_variables["shot_noise"]:
            photons = self.chosen_math_library.random.poisson(readings)
        else:
            photons = readings

        return photons

    def compute_n(self, photoreceptor_rf_size, max_separation=1):
        theta_separation = math.asin(max_separation / self.max_visual_range)
        n = (photoreceptor_rf_size / theta_separation)
        return int(n)
