import numpy as np
import math
import cupy as cp


class FieldOfView:

    def __init__(self, local_dim, max_visual_distance, env_width, env_height):
        self.local_dim = local_dim
        self.max_visual_distance = max_visual_distance
        self.env_width = env_width
        self.env_height = env_height

        self.full_fov_top = None
        self.full_fov_bottom = None
        self.full_fov_left = None
        self.full_fov_right = None

        self.local_fov_top = None
        self.local_fov_bottom = None
        self.local_fov_left = None
        self.local_fov_right = None

        self.enclosed_fov_top = None
        self.enclosed_fov_bottom = None
        self.enclosed_fov_left = None
        self.enclosed_fov_right = None

    def update_field_of_view(self, fish_position):
        fish_position = np.round(fish_position).astype(int)

        self.full_fov_top = fish_position[1] - self.max_visual_distance
        self.full_fov_bottom = fish_position[1] + self.max_visual_distance + 1
        self.full_fov_left = fish_position[0] - self.max_visual_distance
        self.full_fov_right = fish_position[0] + self.max_visual_distance + 1

        self.local_fov_top = 0
        self.local_fov_bottom = self.local_dim
        self.local_fov_left = 0
        self.local_fov_right = self.local_dim

        self.enclosed_fov_top = self.full_fov_top
        self.enclosed_fov_bottom = self.full_fov_bottom
        self.enclosed_fov_left = self.full_fov_left
        self.enclosed_fov_right = self.full_fov_right

        if self.full_fov_top < 0:
            self.enclosed_fov_top = 0
            self.local_fov_top = -self.full_fov_top

        if self.full_fov_bottom > self.env_width:
            self.enclosed_fov_bottom = self.env_width
            self.local_fov_bottom = self.local_dim - (self.full_fov_bottom - self.env_width)

        if self.full_fov_left < 0:
            self.enclosed_fov_left = 0
            self.local_fov_left = -self.full_fov_left

        if self.full_fov_right > self.env_height:
            self.enclosed_fov_right = self.env_height
            self.local_fov_right = self.local_dim - (self.full_fov_right - self.env_height)

class DrawingBoard:
    """Class used to create a 2D image of the environment and surrounding features, and use this to compute photoreceptor
    inputs"""

    def __init__(self, arena_width, arena_height, uv_light_decay_rate, red_light_decay_rate, photoreceptor_rf_size,
                 prey_radius, predator_radius, visible_scatter, dark_light_ratio, dark_gain, light_gain,
                 light_gradient, max_visual_distance, red_object_intensity, red2_object_intensity):

        self.red_object_intensity = red_object_intensity
        self.red2_object_intensity = red2_object_intensity

        self.chosen_math_library = cp

        self.arena_width = arena_width
        self.arena_height = arena_height
        self.uv_light_decay_rate = uv_light_decay_rate
        self.red_light_decay_rate = red_light_decay_rate
        self.light_gain = light_gain
        self.light_gradient = light_gradient
        self.photoreceptor_rf_size = photoreceptor_rf_size

        max_visual_distance = np.round(max_visual_distance).astype(np.int32)
        self.local_dim = max_visual_distance * 2 + 1
        self.max_visual_distance = max_visual_distance
        self.fish_position_FOV = self.chosen_math_library.array([self.max_visual_distance, self.max_visual_distance])

        self.base_db = self.get_base_arena()
        self.base_db_illuminated = self.get_base_arena(visible_scatter)
        self.erase(visible_scatter)
        
        # Repeated computations for sediment
        self.turbPower = 1.0
        self.turbSize = 162.0

        xp, yp = self.chosen_math_library.arange(self.arena_width), self.chosen_math_library.arange(self.arena_height)
        xy, py = self.chosen_math_library.meshgrid(xp, yp)
        xy = self.chosen_math_library.expand_dims(xy, 2)
        py = self.chosen_math_library.expand_dims(py, 2)
        self.coords = self.chosen_math_library.concatenate((xy, py), axis=2)


        self.global_sediment_grating = self.get_sediment()
        self.global_luminance_mask = self.get_luminance_mask(dark_light_ratio, dark_gain)
        self.local_scatter, self.local_scatter_base = self.get_local_scatter()

        self.prey_diameter = prey_radius * 2
        self.prey_radius = prey_radius
        self.predator_size = predator_radius * 2
        self.predator_radius = predator_radius

        self.max_lines_num = 50000

        # Placeholders
#        self.local_db = None
        self.multiplication_matrix = None
        self.addition_matrix = None
        self.mul1_full = None
        self.conditional_tiled = None
        self.mul_for_hypothetical = None
        self.add_for_hypothetical = None
        self.compute_repeated_computations()

        # For debugging purposes
        self.mask_buffer_time_point = None

        # For obstruction mask (reset each time is called).
        self.empty_mask = self.chosen_math_library.ones((self.local_dim, self.local_dim, 1), dtype=np.float64)

        self.FOV = FieldOfView(self.local_dim, self.max_visual_distance, self.arena_width, self.arena_height)

    def get_FOV_size(self):
        return self.local_dim, self.local_dim

    def get_sediment(self):
        xPeriod = self.chosen_math_library.random.uniform(0.0, 10.0)
        yPeriod = self.chosen_math_library.random.uniform(0.0, 10.0)

        noise = self.chosen_math_library.absolute(
            self.chosen_math_library.random.randn(self.arena_width, self.arena_height))

        xy_values = (self.coords[:, :, 0] * xPeriod / self.arena_width) + (self.coords[:, :, 1] * yPeriod / self.arena_height)
        size = self.turbSize

        turbulence = self.chosen_math_library.zeros((self.arena_width, self.arena_height))

        while size >= 1:
            reduced_coords = self.coords / size

            fractX = reduced_coords[:, :, 0] - reduced_coords[:, :, 0].astype(int)
            fractY = reduced_coords[:, :, 1] - reduced_coords[:, :, 1].astype(int)

            x1 = (reduced_coords[:, :, 0].astype(int) + self.arena_width) % self.arena_width
            y1 = (reduced_coords[:, :, 1].astype(int) + self.arena_height) % self.arena_height

            x2 = (x1 + self.arena_width - 1) % self.arena_width
            y2 = (y1 + self.arena_height - 1) % self.arena_height

            value = self.chosen_math_library.zeros((self.arena_width, self.arena_height))
            value += fractX * fractY * noise[y1, x1]
            value += (1 - fractX) * fractY * noise[y1, x2]
            value += fractX * (1 - fractY) * noise[y2, x1]
            value += (1 - fractX) * (1 - fractY) * noise[y2, x2]

            turbulence += value * size
            size /= 2.0

        turbulence = 128 * turbulence / self.turbSize
        xy_values += self.turbPower * turbulence / 256.0
        new_grating = 256 * self.chosen_math_library.abs(self.chosen_math_library.sin(xy_values * 3.14159))
        new_grating /= self.chosen_math_library.max(new_grating)  # Normalise
        new_grating = self.chosen_math_library.expand_dims(new_grating, 2)

        return new_grating * self.red2_object_intensity

    def compute_repeated_computations(self):
        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        self.multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit, (self.max_lines_num, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.local_dim - 1, self.local_dim - 1])
        self.addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (self.max_lines_num, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        self.mul1_full = self.chosen_math_library.tile(mul1, (self.max_lines_num, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.local_dim - 1, self.local_dim - 1, self.local_dim - 1, self.local_dim - 1])
        self.conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (self.max_lines_num, 1))

        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (self.max_lines_num, 1, 1))

        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.local_dim - 1], [self.local_dim - 1, 0]])
        self.add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (self.max_lines_num, 1, 1))

    def compute_repeated_computations_extended(self, num_lines):
        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit, (num_lines, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.local_dim - 1, self.local_dim - 1])
        addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (num_lines, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        mul1_full = self.chosen_math_library.tile(mul1, (num_lines, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.local_dim - 1, self.local_dim - 1, self.local_dim - 1, self.local_dim - 1])
        conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (num_lines, 1))

        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (num_lines, 1, 1))

        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.local_dim - 1], [self.local_dim - 1, 0]])
        add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (num_lines, 1, 1))

        return multiplication_matrix, addition_matrix, mul1_full, mul_for_hypothetical, add_for_hypothetical, conditional_tiled

    def get_local_scatter(self):
        """Computes effects of absorption and scatter, but incorporates effect of implicit scatter from line spread."""

        x, y = self.chosen_math_library.arange(self.local_dim), self.chosen_math_library.arange(self.local_dim)
        y = self.chosen_math_library.expand_dims(y, 1)
        j = self.max_visual_distance + 1
        positional_mask = (((x - j) ** 2 + (y - j) ** 2) ** 0.5)  # Measure of distance from centre to every pixel
        desired_uv_scatter = self.chosen_math_library.exp(-self.uv_light_decay_rate * positional_mask)
        desired_red_scatter = self.chosen_math_library.exp(-self.red_light_decay_rate * positional_mask)
        # To offset the effect of reduced sampling further away from fish (an unwanted effect that yielded massive
        # performance improvements). Should counter them exactly.
        implicit_scatter = self.chosen_math_library.sin(self.photoreceptor_rf_size) * positional_mask
        implicit_scatter[implicit_scatter < 1] = 1

        adjusted_uv_scatter = desired_uv_scatter * implicit_scatter
        adjusted_red_scatter = desired_red_scatter * implicit_scatter
        adjusted_scatter = self.chosen_math_library.stack(
            [adjusted_red_scatter, adjusted_uv_scatter, adjusted_red_scatter],
            axis=2)
        # adjusted_scatter = self.chosen_math_library.expand_dims(adjusted_scatter, 2)
        return adjusted_scatter, desired_uv_scatter

    def create_obstruction_mask_lines_cupy(self, fish_position, prey_locations, predator_locations,
                                           prey_occlusion=False):
        """Must use both numpy and cupy as otherwise GPU runs out of memory. positions are board-centric"""
        # Reset empty mask
        self.empty_mask[:] = 1.0

        # fish position within FOV (central)

        if predator_locations.size == 0 and (not prey_occlusion or prey_locations.size == 0):
            return self.empty_mask

        if prey_occlusion:
            # Compute prey positions already relative to fish (Prey_num, 2)
            prey_relative_positions = prey_locations - fish_position
            # Compute distances of prey from fish.(Prey_num)
            prey_distances = (prey_relative_positions[:, 0] ** 2 + prey_relative_positions[:, 1] ** 2) ** 0.5

            prey_relative_positions = prey_relative_positions[prey_distances < self.max_visual_distance, :]
            prey_distances = prey_distances[prey_distances < self.max_visual_distance]
            prey_locations_FOV = prey_relative_positions + self.max_visual_distance

            # Compute angular size of prey from fish position. (Prey_num)
            prey_half_angular_size = self.chosen_math_library.arctan(self.prey_radius / prey_distances)

            # Compute angle between fish and prey where x-axis=0, and positive values are in upper field. (Prey_num)
            prey_angles = self.chosen_math_library.arctan(prey_relative_positions[:, 1] / prey_relative_positions[:, 0])
            prey_angles = self.chosen_math_library.expand_dims(prey_angles, 1)
            prey_angles = self.chosen_math_library.repeat(prey_angles, 2, 1)

            # From prey angular sizes, compute angles of edges of prey. (Prey_num, 2)
            prey_rf_offsets = self.chosen_math_library.expand_dims(prey_half_angular_size, 1)
            prey_rf_offsets = self.chosen_math_library.repeat(prey_rf_offsets, 2, 1)
            prey_rf_offsets = prey_rf_offsets * self.chosen_math_library.array([-1, 1])
            prey_extremities = prey_angles + prey_rf_offsets

            # Number of lines to project through prey or predators, determined by arena_width, arena_height, and size of features. (1)
            n_lines_prey = self.compute_n(self.chosen_math_library.max(prey_half_angular_size) * 2,
                                          len(prey_locations_FOV))

            # Create array of angles between prey extremities to form lines. (Prey_num * n_lines_prey)
            interpolated_line_angles = self.chosen_math_library.linspace(prey_extremities[:, 0], prey_extremities[:, 1],
                                                                         n_lines_prey).flatten()

            # Computing how far along each line prey are.  (Prey_num * n_lines_prey)
            prey_distance_along = prey_distances + self.prey_radius
            prey_distance_along = self.chosen_math_library.expand_dims(prey_distance_along, 1)
            prey_distance_along = self.chosen_math_library.repeat(prey_distance_along, n_lines_prey, 1)
            prey_distance_along = self.chosen_math_library.swapaxes(prey_distance_along, 0, 1).flatten()

            # Repeat prey locations so matches dimensions of other variables  (Prey_num * n_lines_prey, 2)
            expanded_prey_locations = self.chosen_math_library.tile(prey_locations_FOV, (n_lines_prey, 1))

            # Find which prey are in left half of visual field (as angle convention requires adjustment)
            # (Prey_num * n_lines_prey)
            prey_on_left = (expanded_prey_locations[:, 0] < self.fish_position_FOV[0]) * self.chosen_math_library.pi

            # Assign to array to include predators if present
            distance_along = prey_distance_along
            features_on_left = prey_on_left

        if predator_locations.size != 0:
            # Do the same for predator features.
            predator_relative_positions = predator_locations - fish_position
            predator_locations_FOV = predator_relative_positions + self.max_visual_distance
            predator_distances = (predator_relative_positions[:, 0] ** 2 + predator_relative_positions[:, 1] ** 2) \
                                 ** 0.5
            predator_half_angular_size = self.chosen_math_library.arctan(self.predator_radius / predator_distances)

            predator_angles = self.chosen_math_library.arctan(
                predator_relative_positions[:, 1] / predator_relative_positions[:, 0])
            predator_angles_expanded = self.chosen_math_library.expand_dims(predator_angles, 1)
            predator_angles_expanded = self.chosen_math_library.repeat(predator_angles_expanded, 2, 1)

            predator_rf_offsets = self.chosen_math_library.expand_dims(predator_half_angular_size, 1)
            predator_rf_offsets = self.chosen_math_library.repeat(predator_rf_offsets, 2, 1)
            predator_rf_offsets = predator_rf_offsets * self.chosen_math_library.array([-1, 1])
            predator_extremities = predator_angles_expanded + predator_rf_offsets

            # Number of lines to project through prey or predators, determined by arena_width, arena_height, and size of features.
            n_lines_predator = self.compute_n(self.chosen_math_library.max(predator_half_angular_size) * 2, 1)
            predator_interpolated_line_angles = self.chosen_math_library.linspace(predator_extremities[:, 0],
                                                                                  predator_extremities[:, 1],
                                                                                  n_lines_predator).flatten()

            # Combine prey and predator line angles.
            if prey_occlusion:
                interpolated_line_angles = self.chosen_math_library.concatenate(
                    (interpolated_line_angles, predator_interpolated_line_angles),
                    axis=0)
            else:
                interpolated_line_angles = predator_interpolated_line_angles

            # Computing how far along each line predators are.
            predator_distance_along = (predator_distances ** 2 + self.predator_radius ** 2) ** 0.5
            predator_distance_along = self.chosen_math_library.expand_dims(predator_distance_along, 1)
            predator_distance_along = self.chosen_math_library.repeat(predator_distance_along, int(n_lines_predator), 1)
            predator_distance_along = self.chosen_math_library.swapaxes(predator_distance_along, 0, 1).flatten()
            predator_distance_along = predator_distance_along + self.predator_radius
            if prey_occlusion:
                distance_along = self.chosen_math_library.concatenate((distance_along, predator_distance_along), axis=0)
            else:
                distance_along = predator_distance_along

            expanded_predator_locations = self.chosen_math_library.tile(predator_locations_FOV, (n_lines_predator, 1))
            predators_on_left = (expanded_predator_locations[:, 0] < self.fish_position_FOV[0]) * self.chosen_math_library.pi

            if prey_occlusion:
                features_on_left = self.chosen_math_library.concatenate((features_on_left, predators_on_left), 0)
            else:
                features_on_left = predators_on_left

        # If no predators, and no prey occlusion, return now with empty obstruction mask
        if "interpolated_line_angles" not in locals():
            return self.empty_mask

        total_lines = interpolated_line_angles.shape[0]

        # Ensure all angles are in designated range
        interpolated_line_angles_scaling = (interpolated_line_angles // (
                self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        # Compute m using tan (N_obj x n)
        m = self.chosen_math_library.tan(interpolated_line_angles)

        # Compute c (N_obj*n)
        c = -m * self.fish_position_FOV[0]
        c = c + self.fish_position_FOV[1]

        # Compute components of intersections (N_obj*n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 1)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 1)

        # Slicing repeated matrices so have correct dimensions. (N_obj * n)
        if total_lines > self.max_lines_num:
            print("Too few lines... Attempting to create more.")

            multiplication_matrix = self.multiplication_matrix[:total_lines]
            addition_matrix = self.addition_matrix[:total_lines]
            mul1_full = self.mul1_full[:total_lines]
            mul_for_hypothetical = self.mul_for_hypothetical[:total_lines]
            add_for_hypothetical = self.add_for_hypothetical[:total_lines]
            conditional_tiled = self.conditional_tiled[:total_lines]
        else:
            multiplication_matrix, addition_matrix, mul1_full, mul_for_hypothetical, add_for_hypothetical, \
            conditional_tiled = self.compute_repeated_computations_extended(num_lines=total_lines)

        # Operations to compute all intersections of lines found
        m_mul = self.chosen_math_library.expand_dims(m, 1)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 1)
        m_mul = full_m * mul1_full

        m_mul[:, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, 1] = 1
        division_matrix[:, 3] = 1

        intersection_components = ((c_exp * multiplication_matrix) + addition_matrix) / division_matrix
        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 2)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 2)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (N_obj n 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]

        # Compute angles from the valid intersections (to find those matching original angles) (N_obj x 2)
        possible_vectors = valid_intersection_coordinates - self.fish_position_FOV
        computed_angles = self.chosen_math_library.arctan2(possible_vectors[:, 1], possible_vectors[:, 0])

        # Make sure angles are in correct range.
        angles_scaling = (computed_angles // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        computed_angles = computed_angles + angles_scaling

        computed_angles = self.chosen_math_library.round(computed_angles, 2)

        # Add adjustment for features appearing in left of visual field (needed because of angles convention)
        interpolated_line_angles = interpolated_line_angles + features_on_left

        # Make sure angles in correct range.
        interpolated_line_angles_scaling = (interpolated_line_angles // (
                self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        # Get original angles in correct format.
        original_angles = self.chosen_math_library.round(interpolated_line_angles, 2)
        original_angles = self.chosen_math_library.expand_dims(original_angles, 1)
        original_angles = self.chosen_math_library.repeat(original_angles, 2, 1).flatten()

        # Check which angles match (should be precisely half)
        same_values = (computed_angles == original_angles) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]

        # Finding coordinates of object extremities.
        proj_vector = selected_intersections - self.fish_position_FOV
        proj_distance = (proj_vector[:, 0] ** 2 + proj_vector[:, 1] ** 2) ** 0.5

        # Compute fraction along the projection vector at which features lie
        try:
            fraction_along = distance_along / proj_distance
        except ValueError:
            # Note, think this error happened due to n being halved when more lines were required than were possible...
            print(f"Distance along dimensions: {distance_along.shape}")
            print(f"Proj distance dimensions: {proj_distance.shape}")
            print(f"Fish position: {fish_position}")
            print(f"Prey positions: {prey_locations}")
            print(f"Predator position: {predator_locations}")
            fraction_along = distance_along / proj_distance[:distance_along.shape[0], :]

        fraction_along = self.chosen_math_library.expand_dims(fraction_along, 1)
        fraction_along = self.chosen_math_library.repeat(fraction_along, 2, 1)

        # Find coordinates of lines on features.
        points_on_features = proj_vector * fraction_along
        points_on_features = self.fish_position_FOV + points_on_features
        points_on_features = self.chosen_math_library.expand_dims(points_on_features, 1)

        selected_intersections = self.chosen_math_library.reshape(selected_intersections, (total_lines, 1, 2))

        # Combine the valid wall intersection coordinates with the points found on features (total_lines, 2, 2)
        vertices = self.chosen_math_library.concatenate((selected_intersections, points_on_features), 1)
        vertices_xvals = vertices[:, :, 0]
        vertices_yvals = vertices[:, :, 1]

        # INTERPOLATION  - Get max and min vals in all vertices
        min_x = self.chosen_math_library.min(vertices_xvals, axis=1)
        max_x = self.chosen_math_library.max(vertices_xvals, axis=1)
        min_y = self.chosen_math_library.min(vertices_yvals, axis=1)
        max_y = self.chosen_math_library.max(vertices_yvals, axis=1)

        # SEGMENT COMPUTATION  - Compute length of x and y ranges to test
        try:
            x_lens = self.chosen_math_library.max(max_x - min_x)
        except ValueError:
            # In the event that all prey are further away than max visual distance.
            return self.empty_mask

        y_lens = self.chosen_math_library.max(max_y - min_y)

        x_len = self.chosen_math_library.around(x_lens)
        y_len = self.chosen_math_library.around(y_lens)

        # Get arrays of x and y to interpolate for each line (all must have same length due to array size).
        x_ranges = self.chosen_math_library.linspace(min_x, max_x, int(x_len))
        y_ranges = self.chosen_math_library.linspace(min_y, max_y, int(y_len))

        # Interpolation of lines, both in x and y.
        y_values = (m * x_ranges) + c
        y_values = self.chosen_math_library.floor(y_values)
        set_1 = self.chosen_math_library.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = self.chosen_math_library.floor(x_values)
        set_2 = self.chosen_math_library.stack((x_values, y_ranges), axis=-1)
        full_set = self.chosen_math_library.vstack((set_1, set_2)).astype(int)

        full_set = full_set.reshape(-1, 2)

        try:
            self.empty_mask[full_set[:, 1], full_set[:, 0]] = 0.0
        except IndexError:
            full_set = self.chosen_math_library.clip(full_set, 0, self.local_dim - 1)
            self.empty_mask[full_set[:, 1], full_set[:, 0]] = 0.0

        # For debugging:
        # try:
        #     mask[full_set[:, 1], full_set[:, 0]] = 0  # NOTE: Inverting x and y to match standard in program.
        # except IndexError:
        #     print("IndexError")
        #     full_set[full_set > 1499] = 1499
        #     full_set[full_set < 0] = 0
        #     mask[full_set[:, 1], full_set[:, 0]] = 0  # NOTE: Inverting x and y to match standard in program.
        #
        #     plt.imshow(mask)
        #     plt.scatter(prey_locations[:, 0], prey_locations[:, 1])
        #     plt.show()
        #     mask = None

        occluded = (self.empty_mask[:, :, 0] == 0.0)
        O = self.chosen_math_library.repeat(self.empty_mask, 3, 2)
        # O = self.chosen_math_library.concatenate((self.empty_mask, self.empty_mask, self.empty_mask), axis=2)
        O[:, :, 0] += occluded
        O[:, :, 1] += occluded
        O[:, :, 2] += occluded

        return O

    def get_luminance_mask(self, dark_light_ratio, dark_gain):
        dark_field_length = int(self.arena_width * dark_light_ratio)
        luminance_mask = self.chosen_math_library.ones((self.arena_width, self.arena_height, 1))
        if self.light_gradient > 0 and dark_field_length > 0:
            luminance_mask[:dark_field_length, :, :] *= dark_gain
            luminance_mask[dark_field_length:, :, :] *= self.light_gain
            gradient = self.chosen_math_library.linspace(dark_gain, self.light_gain, self.light_gradient)
            gradient = self.chosen_math_library.expand_dims(gradient, 1)
            gradient = self.chosen_math_library.repeat(gradient, self.arena_height, 1)
            gradient = self.chosen_math_library.expand_dims(gradient, 2)
            luminance_mask[
            int(dark_field_length - (self.light_gradient / 2)):int(dark_field_length + (self.light_gradient / 2)), :,
            :] = gradient

        else:
            luminance_mask[:dark_field_length, :, :] *= dark_gain
            luminance_mask[dark_field_length:, :, :] *= self.light_gain

        return luminance_mask

    def extend_A(self, A):
        """Extends the arena pixels (red1 channel), by stretching the values computed at the wall points along the
        whole FOV in that direction"""

        if self.FOV.full_fov_top < 0:
            low_dim_top = abs(self.FOV.full_fov_top)
        else:
            low_dim_top = 0
        if self.FOV.full_fov_left < 0:
            low_dim_left = abs(self.FOV.full_fov_left)
        else:
            low_dim_left = 0

        if self.FOV.full_fov_bottom > self.arena_height:
            high_dim_bottom = abs(self.FOV.full_fov_bottom) - (self.arena_height - 1)
        else:
            high_dim_bottom = 0
        if self.FOV.full_fov_right > self.arena_width:
            high_dim_right = abs(self.FOV.full_fov_right) - (self.arena_width - 1)
        else:
            high_dim_right = 0

        # Top and left walls
        pixel_strip_x = A[:, low_dim_left, 0]  # Preserve luminance gradient by taking slice.
        pixel_block_x = self.chosen_math_library.repeat(self.chosen_math_library.expand_dims(pixel_strip_x, 1),
                                                        low_dim_left, axis=1)
        A[:, :low_dim_left, 0] = pixel_block_x

        pixel_strip_y = A[low_dim_top, :, 0]
        pixel_block_y = self.chosen_math_library.repeat(self.chosen_math_library.expand_dims(pixel_strip_y, 0),
                                                        low_dim_top, axis=0)
        A[:low_dim_top, :, 0] = pixel_block_y

        # Bottom and right walls
        pixel_strip_x = A[:, -high_dim_right, 0]
        pixel_block_x = self.chosen_math_library.repeat(self.chosen_math_library.expand_dims(pixel_strip_x, 1),
                                                        high_dim_right, 1)
        if high_dim_right == 0:  # Necessary due to numpy indexing inconsistency.
            high_dim_right = -A.shape[1]
        A[:, -high_dim_right:, 0] = pixel_block_x

        pixel_strip_y = A[-high_dim_bottom, :, 0]
        pixel_block_y = self.chosen_math_library.repeat(self.chosen_math_library.expand_dims(pixel_strip_y, 0),
                                                        high_dim_bottom, 0)
        if high_dim_bottom == 0:
            high_dim_bottom = -A.shape[0]
        A[-high_dim_bottom:, :, 0] = pixel_block_y

        return A

    def get_masked_pixels(self, fish_position, prey_locations, predator_locations):
        """
        Returns masked pixels in form W.H.3
        With Red.UV.Red2
        """
        try:
            A = self.chosen_math_library.array(self.local_db)
        except ValueError:
            print(self.local_db)

        # apply FOV portion of luminance mask
        local_luminance_mask = self.chosen_math_library.zeros(self.local_db.shape)
        lum_slice = self.global_luminance_mask[self.FOV.enclosed_fov_top:self.FOV.enclosed_fov_bottom,
                                               self.FOV.enclosed_fov_left:self.FOV.enclosed_fov_right, :]
        local_luminance_mask[self.FOV.local_fov_top:self.FOV.local_fov_bottom,
                             self.FOV.local_fov_left:self.FOV.local_fov_right] = lum_slice

        A *= local_luminance_mask

        # If FOV extends outside the arena, extend the A image
        A = self.extend_A(A)

        if prey_locations.size + predator_locations.size == 0:
            O = self.chosen_math_library.ones((self.local_dim, self.local_dim, 3), dtype=np.float64)
        else:
            O = self.create_obstruction_mask_lines_cupy(self.chosen_math_library.array(fish_position),
                                                        self.chosen_math_library.array(prey_locations),
                                                        self.chosen_math_library.array(predator_locations))

        return A * O * self.local_scatter, local_luminance_mask[:, :, 1] * self.local_scatter_base

    def compute_n(self, angular_size, number_of_this_feature, max_separation=1):
        """Computes the number of lines from which the visual input ot a photoreceptor is evaluated."""

        theta_separation = math.asin(max_separation / self.max_visual_distance)
        n = (angular_size / theta_separation)

        if n * number_of_this_feature > self.max_lines_num:
            print(f"""Max lines num needs increase:
            Max lines num: {self.max_lines_num}
            Required lines for this feature alone: {n * number_of_this_feature}
            """)
            n = (self.max_lines_num * 0.8) / number_of_this_feature

        return int(n)

    def reset(self):
        """To be called at start of episode"""
        self.global_sediment_grating = self.get_sediment()

    def erase(self, bkg=0.):
        if bkg == 0.:
            self.local_db = self.chosen_math_library.copy(self.base_db)
        else:
            self.local_db = self.chosen_math_library.copy(self.base_db_illuminated)

    def get_base_arena(self, bkg=0.0):
        if bkg == 0:
            db = self.chosen_math_library.zeros((self.local_dim, self.local_dim, 3), dtype=np.double)
        else:
            db = (self.chosen_math_library.ones((self.local_dim, self.local_dim, 3), dtype=np.double) * bkg)
        return db

    def draw_sediment(self):
        """Slice the global sediment for current field of view"""

        sediment_slice = self.global_sediment_grating[self.FOV.enclosed_fov_top:self.FOV.enclosed_fov_bottom,
                                                      self.FOV.enclosed_fov_left:self.FOV.enclosed_fov_right, 0]

        self.local_db[self.FOV.local_fov_top:self.FOV.local_fov_bottom,
                      self.FOV.local_fov_left:self.FOV.local_fov_right, 2] = sediment_slice

    def draw_walls(self):
        """Draws walls as deep into FOV beyond wall objects as possible."""

        self.local_db[self.FOV.local_fov_top, self.FOV.local_fov_left:self.FOV.local_fov_right, 0] = self.red_object_intensity

        self.local_db[self.FOV.local_fov_bottom - 1, self.FOV.local_fov_left:self.FOV.local_fov_right, 0] = self.red_object_intensity

        self.local_db[self.FOV.local_fov_top:self.FOV.local_fov_bottom, self.FOV.local_fov_left, 0] = self.red_object_intensity

        self.local_db[self.FOV.local_fov_top:self.FOV.local_fov_bottom, self.FOV.local_fov_right - 1, 0] = self.red_object_intensity
