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
from geometry import FieldOfView

class Arena:
    """Class used to create a 2D image of the environment and surrounding features, and use this to compute photoreceptor
    inputs"""

    def __init__(self, env_variables, rng):
        self.test_mode = env_variables['test_sensory_system']
        self.rng = rng
        self.bottom_intensity = env_variables['bottom_intensity']
        self.max_uv_range = np.round(np.absolute(np.log(0.001) / env_variables["light_decay_rate"])).astype(np.int32)

        self.arena_width = env_variables['arena_width']
        self.arena_height = env_variables['arena_height']
        self.dark_gain = env_variables['dark_gain']
        self.light_gradient = env_variables['light_gradient']
        if self.test_mode:
            self.dark_light_ratio = 0.5
        else:
            self.dark_light_ratio = env_variables['dark_light_ratio']

        self.sediment_sigma = env_variables['sediment_sigma']

        max_viewing_elevation = max(env_variables["viewing_elevations"])
        self.max_red_range = np.round(env_variables["elevation"] * np.tan(np.radians(max_viewing_elevation))).astype(np.int32) + 8 # +8 is to allow for some extra space around the fish in FOV


        self.global_sediment_grating = self.get_global_sediment()
        self.global_luminance_mask = self.get_global_luminance()
        self.illuminated_sediment = self.global_sediment_grating * self.global_luminance_mask

        self.red_FOV = FieldOfView(self.max_red_range, self.arena_width, self.arena_height, env_variables['light_decay_rate'])
        self.uv_FOV = FieldOfView(self.max_uv_range, self.arena_width, self.arena_height, env_variables['light_decay_rate'])

    def get_global_sediment(self):

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
        dark_field_length = int(self.arena_height * self.dark_light_ratio)
        luminance_mask = np.ones((self.arena_width, self.arena_height))
        if self.light_gradient > 0 and dark_field_length > 0:
            luminance_mask[:dark_field_length, :] *= self.dark_gain
            luminance_mask[dark_field_length:, :] *= 1
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
        """Returns the sediment grating masked by the luminance mask"""
        return self.red_FOV.get_sliced_masked_image(self.illuminated_sediment)

    def get_uv_luminance_mask(self):
        """Returns the luminance mask for the UV field of view"""
        return self.uv_FOV.get_sliced_masked_image(self.global_luminance_mask)
    
    def reset(self):
        """To be called at start of episode"""
        self.global_sediment_grating = self.get_global_sediment()
        self.illuminated_sediment = self.global_sediment_grating * self.global_luminance_mask

