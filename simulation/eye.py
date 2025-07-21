# Copyright 2025 Asaph Zylbertal & Sam Pink
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import geometry

class Eye:

    def __init__(self, verg_angle, retinal_field, is_left, env_variables, max_uv_range,rng):

        self.test_mode = env_variables["test_sensory_system"]
        self.rng = rng
        self.viewing_elevations = env_variables["viewing_elevations"]
        self.fish_elevation = env_variables["elevation"]
        self.uv_object_intensity = env_variables["uv_object_intensity"]
        self.water_uv_scatter = env_variables["water_uv_scatter"]
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

        self.uv_photoreceptor_rf_size = env_variables['uv_photoreceptor_rf_size']
        self.red_photoreceptor_rf_size = env_variables['red_photoreceptor_rf_size']

        self.uv_photoreceptor_angles = self.update_angles_sigmoid(verg_angle, retinal_field, is_left)
        self.uv_photoreceptor_num = len(self.uv_photoreceptor_angles)
        self.red_photoreceptor_num = self.uv_photoreceptor_num

        self.interpolated_observation_angles = np.arange(
            np.min(self.uv_photoreceptor_angles),
            np.max(self.uv_photoreceptor_angles) + self.sz_rf_spacing / 2,
            self.sz_rf_spacing / 2)

        self.red_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left,
                                                           self.red_photoreceptor_num)

        self.total_photoreceptor_num = self.uv_photoreceptor_num + self.red_photoreceptor_num

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

    def read(self, masked_arena_pixels, eye_x, eye_y, fish_angle, uv_lum_mask, prey_positions, predator_left, predator_right, predator_dist,
             proj=True):
        """
        Resolve RF coordinates for each photoreceptor, and use those to sum the relevant pixels.
        """

        corrected_uv_pr_angles = np.arctan2(np.sin(self.uv_photoreceptor_angles + fish_angle),
                                                                np.cos(self.uv_photoreceptor_angles + fish_angle))
        eye_FOV_x = eye_x + (uv_lum_mask.shape[1] - 1) / 2
        eye_FOV_y = eye_y + (uv_lum_mask.shape[0] - 1) / 2

        uv_readings = self.water_uv_scatter * np.expand_dims(geometry.ray_sum(uv_lum_mask, (eye_FOV_y, eye_FOV_x), corrected_uv_pr_angles), axis=1)
        red_readings = np.zeros((self.red_photoreceptor_num, 2))
        for ii, view_elevation in enumerate(self.viewing_elevations):
            red_readings[:, ii] = self._read_elevation(masked_arena_pixels, eye_x, eye_y, view_elevation, fish_angle,
                                 self.red_photoreceptor_angles, self.red_photoreceptor_rf_size, predator_left, predator_right, predator_dist)

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

            uv_readings += proj_uv_readings

        uv_readings = self.add_noise_to_readings(uv_readings)
        red_readings = self.add_noise_to_readings(red_readings)

        interp_uv_readings = np.zeros((self.interpolated_observation_angles.shape[0], 1))
        interp_red_readings = np.zeros((self.interpolated_observation_angles.shape[0], 2))
        interp_uv_readings[:, 0] = np.interp(self.interpolated_observation_angles,
                                                                   self.uv_photoreceptor_angles, uv_readings[:, 0])
        interp_red_readings[:, 0] = np.interp(self.interpolated_observation_angles,
                                                                    self.red_photoreceptor_angles,
                                                                    red_readings[:, 0])
        interp_red_readings[:, 1] = np.interp(self.interpolated_observation_angles,
                                                                    self.red_photoreceptor_angles,
                                                                    red_readings[:, 1])

        readings = np.concatenate((interp_uv_readings, interp_red_readings), axis=1)

        # Rounds down observations to form array of discrete photon events
        photons = np.floor(readings).astype(int)
        photons = photons.clip(0, 255)

        return photons

    def _read_elevation(self, masked_pixels, eye_x, eye_y, elevation_angle, fish_angle, pr_angles, rf_size, predator_left, predator_right, predator_dist):


        eye_FOV_x = int(eye_x + (masked_pixels.shape[1] - 1) / 2)
        eye_FOV_y = int(eye_y + (masked_pixels.shape[0] - 1) / 2)
        radius = int(np.round(self.fish_elevation * np.tan(np.radians(elevation_angle))))
        circle_values, circle_angles = geometry.circle_edge_pixels(masked_pixels, (eye_FOV_y, eye_FOV_x), radius)
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
        ang_bin = 0.001  # this is the bin size for the projection
        proj_angles = np.arange(-np.pi, np.pi + ang_bin, ang_bin)  # this is the angle range for the projection

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

        l_ind = self._closest_index_parallel(proj_angles, theta - half_angle).astype(int)
        r_ind = self._closest_index_parallel(proj_angles, theta + half_angle).astype(int)

        prey_brightness = lum_mask[(np.floor(prey_pos_in_range[:, 1]) - 1).astype(int),
                                   (np.floor(prey_pos_in_range[:, 0]) - 1).astype(
                                       int)]  # includes absorption (???)

        proj = np.zeros((p_num, len(proj_angles)))

        prey_brightness = np.expand_dims(prey_brightness, 1)

        r = np.arange(proj.shape[1])
        prey_present = (l_ind[:, None] <= r) & (r_ind[:, None] >= r)
        prey_present = prey_present.astype(float)
        prey_present *= prey_brightness

        total_angular_input = np.sum(prey_present, axis=0)

        pr_ind_s = self._closest_index_parallel(proj_angles, uv_pr_angles - rf_size / 2)
        pr_ind_e = self._closest_index_parallel(proj_angles, uv_pr_angles + rf_size / 2)

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
            photons = self.rng.poisson(readings)
        else:
            photons = readings

        return photons
