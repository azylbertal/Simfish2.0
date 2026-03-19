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
from scipy.ndimage import gaussian_filter

class FieldOfView:
    """

    This class handles the calculation and management of a fish's visual perception area,
    including light absorption/scattering effects based on distance. It computes the visible region
    around a fish's position and applies distance-based light decay.
    Attributes:
        local_dim (int): The dimension of the local field of view (2 * max_range + 1).
        max_visual_distance (int): The maximum distance the fish can see (rounded).
        env_width (int): The width of the environment.
        env_height (int): The height of the environment.
        light_decay_rate (float): The rate at which light decays with distance.
        local_scatter (np.ndarray): Precomputed light decay mask based on distance.
        full_fov_top (int): Top boundary of the full FOV in environment coordinates.
        full_fov_bottom (int): Bottom boundary of the full FOV in environment coordinates.
        full_fov_left (int): Left boundary of the full FOV in environment coordinates.
        full_fov_right (int): Right boundary of the full FOV in environment coordinates.
        local_fov_top (int): Top boundary in local FOV coordinates.
        local_fov_bottom (int): Bottom boundary in local FOV coordinates.
        local_fov_left (int): Left boundary in local FOV coordinates.
        local_fov_right (int): Right boundary in local FOV coordinates.
        enclosed_fov_top (int): Top boundary of FOV enclosed within environment bounds.
        enclosed_fov_bottom (int): Bottom boundary of FOV enclosed within environment bounds.
        enclosed_fov_left (int): Left boundary of FOV enclosed within environment bounds.
        enclosed_fov_right (int): Right boundary of FOV enclosed within environment bounds.
    Methods:
        _get_local_scatter(): Computes the light decay mask based on distance from center.
        update_field_of_view(fish_position): Updates FOV boundaries based on fish position.
        get_sliced_masked_image(img): Extracts and applies light decay to the visible region.
    """

    def __init__(self, max_range, env_width, env_height, light_decay_rate):
        round_max_range = int(np.round(max_range))
        local_dim = round_max_range * 2 + 1
        self.local_dim = local_dim
        self.max_visual_distance = round_max_range
        self.env_width = env_width
        self.env_height = env_height
        self.light_decay_rate = light_decay_rate
        self.local_scatter = self._get_local_scatter()
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

    def _get_local_scatter(self):
        """Computes effects of absorption and scatter"""
        x, y = np.arange(self.local_dim), np.arange(self.local_dim)
        y = np.expand_dims(y, 1)
        j = self.max_visual_distance# + 1
        positional_mask = (((x - j) ** 2 + (y - j) ** 2) ** 0.5)  # Measure of distance from centre to every pixel
        return np.exp(-self.light_decay_rate * positional_mask)

    def update_field_of_view(self, fish_position):
        """
        Updates the field of view (FOV) boundaries based on the fish's position.
        """
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

    def get_sliced_masked_image(self, img):
        """
        Extracts the relevant portion of the image based on the current FOV and applies the light decay mask.
        """

        # apply FOV portion of luminance mask
        masked_image = np.zeros((self.local_dim, self.local_dim))

        slice = img[self.enclosed_fov_top:self.enclosed_fov_bottom,
                    self.enclosed_fov_left:self.enclosed_fov_right]
        
        masked_image[self.local_fov_top:self.local_fov_bottom,
                             self.local_fov_left:self.local_fov_right] = slice


        return masked_image * self.local_scatter
    
class Arena:
    """
    Simulated environment for fish behavior experiments.

    Manages the visual environment including sediment patterns, luminance 
    gradients, and spectral fields of view (Red and UV). It defines the 
    spatial and lighting conditions perceived by agents during simulation.

    Parameters
    ----------
    env_variables : dict
        Dictionary containing configuration parameters such as 'arena_width', 
        'arena_height', 'arena_light_decay_rate', and 'arena_sediment_sigma'.
    rng : np.random.Generator
        Random number generator for stochastic sediment pattern generation.

    Attributes
    ----------
    test_mode : bool
        True if the arena is in sensory system testing mode (fixed patterns).
    arena_width : int
        Width of the arena in pixels.
    arena_height : int
        Height of the arena in pixels.
    bottom_intensity : float
        Base intensity value for the sediment pattern.
    dark_gain : float
        Intensity multiplier for the dark region of the arena.
    light_gradient : int
        Width in pixels of the transition zone between dark and light regions.
    dark_light_ratio : float
        Fraction of arena height that is dark (0.0 to 1.0).
    sediment_sigma : float
        Gaussian filter sigma for smoothing the stochastic sediment patterns.
    max_uv_range : int
        Maximum range in pixels for UV field of view based on light decay.
    max_red_range : int
        Maximum range in pixels for red field of view based on viewing angle.
    global_sediment_grating : np.ndarray
        The underlying sediment texture (2D array) before lighting is applied.
    global_luminance_mask : np.ndarray
        The 2D spatial mask representing light and dark regions.
    illuminated_sediment : np.ndarray
        The final visual output (sediment * luminance).
    red_FOV : FieldOfView
        Manager for the red channel spectral field of view.
    uv_FOV : FieldOfView
        Manager for the UV channel spectral field of view.
    """


    def __init__(self, env_variables, rng):
        self.test_mode = env_variables['test_sensory_system']
        self.rng = rng
        self.bottom_intensity = env_variables['arena_bottom_intensity']
        self.max_uv_range = np.round(np.absolute(np.log(0.001) / env_variables["arena_light_decay_rate"])).astype(np.int32)

        self.arena_width = env_variables['arena_width']
        self.arena_height = env_variables['arena_height']
        self.dark_gain = env_variables['arena_dark_gain']
        self.light_gradient = env_variables['arena_light_gradient']
        if self.test_mode:
            self.dark_light_ratio = 0.5
        else:
            self.dark_light_ratio = env_variables['arena_dark_fraction']

        self.sediment_sigma = env_variables['arena_sediment_sigma']

        max_viewing_elevation = max(env_variables["eyes_viewing_elevations"])
        self.max_red_range = np.round(env_variables["fish_elevation"] * np.tan(np.radians(max_viewing_elevation))).astype(np.int32) + 8 # +8 is to allow for some extra space around the fish in FOV


        self.global_sediment_grating = self.get_global_sediment()
        self.global_luminance_mask = self.get_global_luminance()
        self.illuminated_sediment = self.global_sediment_grating * self.global_luminance_mask

        self.red_FOV = FieldOfView(self.max_red_range, self.arena_width, self.arena_height, env_variables['arena_light_decay_rate'])
        self.uv_FOV = FieldOfView(self.max_uv_range, self.arena_width, self.arena_height, env_variables['arena_light_decay_rate'])

    def get_global_sediment(self):
        """
        Generate the base sediment pattern for the entire arena.

        In test mode, this produces a deterministic vertical grating. In 
        normal mode, it produces smoothed Gaussian noise.

        Returns
        -------
        np.ndarray
            2D array representing the sediment intensity across the arena.
        """
        if self.test_mode:
            # In test mode, use a fixed grating for testing purposes
            new_grating = np.zeros((self.arena_width, self.arena_height))
            # draw vertical stripes, 10 pixels in width
            for i in range(0, self.arena_width, 20):
                new_grating[:, i:i+10] = 1.0
            new_grating[self.arena_height // 2 - 80: self.arena_height // 2 - 30, self.arena_width // 2 + 30: self.arena_width // 2 + 80] = 0
            new_grating[self.arena_height // 2 + 40: self.arena_height // 2 + 120, self.arena_width // 2 + 20: self.arena_width // 2 + 100] = 10.

        else:
            new_grating = self.rng.random((self.arena_width, self.arena_height))
            new_grating = gaussian_filter(new_grating, sigma=self.sediment_sigma)
            new_grating -= np.min(new_grating)
            new_grating /= np.max(new_grating)

        return new_grating * self.bottom_intensity

    def get_global_luminance(self):
        """
        Create the luminance gradient mask for the arena.

        Calculates the transition between dark and light fields based on the 
        `dark_light_ratio` and `light_gradient` width.

        Returns
        -------
        np.ndarray
            2D array mask of intensity multipliers.
        """
        dark_field_length = int(self.arena_height * self.dark_light_ratio)
        luminance_mask = np.ones((self.arena_width, self.arena_height))
        if self.light_gradient > 0 and dark_field_length > 0:
            luminance_mask[:dark_field_length, :] *= self.dark_gain
            gradient = np.linspace(self.dark_gain, 1, self.light_gradient)
            gradient = np.expand_dims(gradient, 1)
            gradient = np.repeat(gradient, self.arena_height, 1)
            #gradient = np.expand_dims(gradient, 2)
            luminance_mask[int(dark_field_length - (self.light_gradient / 2)):int(dark_field_length + (self.light_gradient / 2)), :] = gradient

        else:
            luminance_mask[:dark_field_length, :] *= self.dark_gain
            luminance_mask[dark_field_length:, :] *= 1

        return luminance_mask


    def get_masked_sediment(self):
        """
        Retrieve the sediment visible within the red field of view.

        Returns
        -------
        np.ndarray
            Sliced and masked image corresponding to the red FOV.
        """
        return self.red_FOV.get_sliced_masked_image(self.illuminated_sediment)

    def get_uv_luminance_mask(self):
        """
        Retrieve the luminance mask within the UV field of view.

        Returns
        -------
        np.ndarray
            Sliced and masked image corresponding to the UV FOV.
        """
        return self.uv_FOV.get_sliced_masked_image(self.global_luminance_mask)
    
    def reset(self):
        """
        Regenerate the sediment pattern for a new simulation episode.
        
        Resets the `global_sediment_grating` and updates `illuminated_sediment`.
        """
        self.global_sediment_grating = self.get_global_sediment()
        self.illuminated_sediment = self.global_sediment_grating * self.global_luminance_mask

