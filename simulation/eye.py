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
import simulation.geometry as geometry

class Eye:

    def __init__(self, verg_angle, retinal_field, is_left, env_variables, max_uv_range,rng):

        self.test_mode = env_variables["test_sensory_system"]
        self.rng = rng
        self.viewing_elevations = env_variables["eyes_viewing_elevations"]
        self.fish_elevation = env_variables["fish_elevation"]
        self.uv_object_intensity = env_variables["arena_uv_object_intensity"]
        self.water_uv_scatter = env_variables["arena_water_uv_scatter"]
        self.retinal_field_size = retinal_field
        self.env_variables = env_variables
        self.max_uv_range = max_uv_range
        self.prey_diameter = self.env_variables['prey_radius'] * 2

        self.sz_rf_spacing = self.env_variables["eyes_sz_rf_spacing"]
        self.sz_size = self.env_variables["eyes_sz_size"]
        self.sz_oversampling_factor = self.env_variables["eyes_sz_oversampling_factor"]
        self.sigmoid_steepness = self.env_variables["eyes_sz_edge_steepness"]

        self.periphery_rf_spacing = self.sz_rf_spacing * self.sz_oversampling_factor
        self.density_range = self.periphery_rf_spacing - self.sz_rf_spacing

        self.uv_photoreceptor_rf_size = env_variables['eyes_uv_photoreceptor_rf_size']
        self.red_photoreceptor_rf_size = env_variables['eyes_red_photoreceptor_rf_size']

        self.uv_photoreceptor_angles = self.update_angles_sigmoid(verg_angle, retinal_field, is_left)
        self.channel_photoreceptor_num = len(self.uv_photoreceptor_angles)

        self.interpolated_observation_angles = np.arange(
            np.min(self.uv_photoreceptor_angles),
            np.max(self.uv_photoreceptor_angles) + self.sz_rf_spacing / 2,
            self.sz_rf_spacing / 2)

        self.red_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left,
                                                           self.channel_photoreceptor_num)


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
        red_readings = np.zeros((self.channel_photoreceptor_num, 2))
        for ii, view_elevation in enumerate(self.viewing_elevations):
            red_readings[:, ii] = geometry.read_elevation(self.fish_elevation, masked_arena_pixels, eye_x, eye_y, view_elevation, fish_angle,
                                 self.red_photoreceptor_angles, self.red_photoreceptor_rf_size, predator_left, predator_right, predator_dist)

        uv_items = prey_positions
        if proj and (len(uv_items)) > 0:
            proj_uv_readings = geometry.read_prey_proj(max_uv_range=self.max_uv_range,
                                                       prey_diameter=self.prey_diameter,
                                                       eye_x=eye_x,
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

    def add_noise_to_readings(self, readings):
        """Samples from Poisson distribution to get number of photons"""
        if self.env_variables["eyes_shot_noise"]:
            photons = self.rng.poisson(readings)
        else:
            photons = readings

        return photons
