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
        """
        Initialize an Eye object with specified parameters for simulating fish vision.
        Args:
            verg_angle: The vergence angle of the eye (angle at which the eye is rotated inwards).
            retinal_field: The size of the retinal field of view.
            is_left: Boolean indicating if this is the left eye (True) or right eye (False).
            env_variables: Dictionary containing environment configuration parameters including:
                - test_sensory_system: Flag for testing mode
                - eyes_viewing_elevations: Elevation angles for viewing the bottom sediment
                - fish_elevation: Elevation position of the fish above the bottom
                - arena_uv_object_intensity: UV scattering by objects in arena
                - arena_water_uv_scatter: UV scattering by water
                - prey_radius: Radius of prey objects
                - eyes_sz_rf_spacing: spacing of receptive fields in the strike zone
                - eyes_sz_size: Size of the strike zone
                - eyes_sz_oversampling_factor: Oversampling factor for strike zone relative to the rest of the periphery
                - eyes_sz_edge_steepness: Steepness of sigmoid edge function
                - eyes_uv_photoreceptor_rf_size: Angular size of UV photoreceptor receptive fields
                - eyes_red_photoreceptor_rf_size: Angular size of red photoreceptor receptive fields
            max_uv_range: Maximum range for UV detection (usually determined by the light decay rate parameter)
            rng: Random number generator instance for simulation randomness.

        """

        self.test_mode = env_variables["test_sensory_system"]
        self.rng_p = np.random.default_rng(seed=rng.integers(0, 10000)) # Random number generator for the shot noise, to avoid different mean accounts affecting the generator position
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

        self.uv_photoreceptor_angles = self._update_angles_sigmoid(verg_angle, retinal_field, is_left)
        self.channel_photoreceptor_num = len(self.uv_photoreceptor_angles)

        self.interpolated_observation_angles = np.arange(
            np.min(self.uv_photoreceptor_angles),
            np.max(self.uv_photoreceptor_angles) + self.sz_rf_spacing / 2,
            self.sz_rf_spacing / 2)

        self.red_photoreceptor_angles = self._update_angles_linear(verg_angle, retinal_field, is_left,
                                                           self.channel_photoreceptor_num)


    def _update_angles_sigmoid(self, verg_angle, retinal_field, is_left):
        """
        Calculate the angular positions of photoreceptors on the retina using a sigmoid-based spacing function.
        This method generates photoreceptor positions with variable spacing that changes according to
        a sigmoid function, creating higher density in certain regions of the retina. The positions
        are then transformed to angular coordinates in the eye's reference frame.
        Parameters
        ----------
        verg_angle : float
            The vergence angle of the eyes (in radians). Positive values indicate convergence.
        retinal_field : float
            The total angular extent of the retinal field (in radians).
        is_left : bool
            True if calculating for the left eye, False for the right eye.
        Returns
        -------
        np.ndarray
            Sorted array of angular positions (in radians) for photoreceptors in the eye's
            reference frame. For the left eye, angles decrease from the maximum angle.
            For the right eye, angles increase from the minimum angle.
        Notes
        -----
        The spacing between photoreceptors is determined by:
            spacing = sz_rf_spacing + density_range / (1 + exp(-sigmoid_steepness * (pr - sz_size)))
        where the sigmoid function creates a smooth transition in receptor density across the retina.
        """

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

    def _update_angles_linear(self, verg_angle, retinal_field, is_left, photoreceptor_num):
        """
        Calculate photoreceptor angles with linear spacing for an eye.

        This method computes the angular positions of photoreceptors across the retinal field,
        taking into account eye vergence (convergence/divergence) and whether it's the left or
        right eye.

        Args:
            verg_angle (float): The vergence angle in radians. Positive values indicate 
                convergence (eyes turning inward), negative values indicate divergence.
            retinal_field (float): The total angular field of view covered by the retina 
                in radians.
            is_left (bool): True if calculating angles for the left eye, False for the 
                right eye.
            photoreceptor_num (int): The number of photoreceptors to distribute across 
                the retinal field.

        Returns:
            numpy.ndarray: Array of angles in radians representing the angular position 
                of each photoreceptor. For the left eye, angles are centered around 
                -π/2 (pointing left), and for the right eye, around π/2 (pointing right).

        Notes:
            - Left eye angles are centered at -π/2 with positive vergence shifting angles 
              toward the center (rightward).
            - Right eye angles are centered at π/2 with positive vergence shifting angles 
              toward the center (leftward).
        """

        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        return np.linspace(min_angle, max_angle, photoreceptor_num)

    def read(self, masked_arena_pixels, eye_x, eye_y, fish_angle, uv_lum_mask, prey_positions, predator_left, predator_right, predator_dist):
        """
        Read and process visual input from one simulated fish eye.
        This method simulates the visual system of a fish by computing photoreceptor readings
        from UV and red channels at different viewing elevations, applying noise, interpolating
        the readings, and converting them to discrete photon counts.
        Args:
            masked_arena_pixels: Masked pixel array of the sediment environment
            eye_x (float): X-coordinate of the eye position
            eye_y (float): Y-coordinate of the eye position
            fish_angle (float): Current orientation angle of the fish in radians
            uv_lum_mask: UV luminance mask array representing the UV light distribution around the fish
            prey_positions: Array of prey positions for UV object detection
            predator_left: angle of the left edge of the predator
            predator_right: angle of the right edge of the predator
            predator_dist: Distance to predator
        Returns:
            numpy.ndarray: Array of shape (num_interpolated_angles, 3) containing discrete photon
                           counts clipped to range [0, 255]. Columns represent:
                           - Column 0: UV channel readings
                           - Column 1: Red channel readings at first viewing elevation
                           - Column 2: Red channel readings at second viewing elevation
        Notes:
            - UV readings incorporate water scatter effects and optional prey projections
            - Red readings are computed for multiple viewing elevations
            - Noise is added to both UV and red readings
            - Final readings are interpolated to a standard set of observation angles
            - Photon counts are discretized by flooring and clipping to valid range
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
        if len(uv_items) > 0:
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
        """
        Add shot noise to photon readings using Poisson distribution.
        This method simulates the quantum nature of light by sampling from a Poisson
        distribution, which models the statistical fluctuations inherent in photon
        detection (shot noise). If shot noise is disabled in environment variables,
        the readings are returned unchanged.
        Parameters
        ----------
        readings : numpy.ndarray or float
            The input photon readings or expected photon counts to which noise
            will be added.
        Returns
        -------
        numpy.ndarray or float
            The noisy photon counts sampled from a Poisson distribution with
            lambda parameter equal to the input readings. If shot noise is
            disabled, returns the original readings unchanged.
        Notes
        -----
        Shot noise is enabled/disabled via the `eyes->shot_noise` field in the
        env_config yaml file. The Poisson sampling uses the instance's
        random number generator `rng_p` to ensure reproducibility.
        """
        """Samples from Poisson distribution to get number of photons"""
        if self.env_variables["eyes_shot_noise"]:
            photons = self.rng_p.poisson(readings)
        else:
            photons = readings

        return photons
