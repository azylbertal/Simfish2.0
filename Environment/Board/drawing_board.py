import numpy as np
import math
#import cupy as cp
from scipy.ndimage import gaussian_filter

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

    # def __init__(self, arena_width, arena_height, uv_light_decay_rate, red_light_decay_rate, photoreceptor_rf_size,
    #              prey_radius, predator_radius, visible_scatter, dark_light_ratio, dark_gain, light_gain,
    #              light_gradient, max_red_range, max_uv_range, red_object_intensity, sediment_sigma):#, red2_object_intensity):
    def __init__(self, env_variables):#, red2_object_intensity):
        self.test_mode = env_variables['test_sensory_system']

        self.bottom_intensity = env_variables['bottom_intensity']
        #self.red2_object_intensity = red2_object_intensity
        self.max_uv_range = np.round(np.absolute(np.log(0.001) / env_variables["light_decay_rate"])).astype(np.int32)

        self.arena_width = env_variables['arena_width']
        self.arena_height = env_variables['arena_height']
        self.uv_light_decay_rate = env_variables['light_decay_rate']
        self.red_light_decay_rate = env_variables['light_decay_rate']
        self.dark_gain = env_variables['dark_gain']
        self.light_gradient = env_variables['light_gradient']
        if self.test_mode:
            self.dark_light_ratio = 0.5
        else:
            self.dark_light_ratio = env_variables['dark_light_ratio']

        self.photoreceptor_rf_size = max([env_variables['uv_photoreceptor_rf_size'],
                                         env_variables['red_photoreceptor_rf_size']])

        self.sediment_sigma = env_variables['sediment_sigma']

        max_viewing_elevation = max(env_variables["viewing_elevations"])
        self.max_red_range = np.round(env_variables["elevation"] * np.tan(np.radians(max_viewing_elevation))).astype(np.int32) + 8 # +8 is to allow for some extra space around the fish in FOV


                
        self.local_dim = self.max_red_range * 2 + 1
                
        self.fish_position_FOV = np.array([self.max_red_range, self.max_red_range])

        self.base_db = np.zeros((self.local_dim, self.local_dim), dtype=np.double)
        self.erase()
        
        # xp, yp = np.arange(self.arena_width), np.arange(self.arena_height)
        # xy, py = np.meshgrid(xp, yp)
        # xy = np.expand_dims(xy, 2)
        # py = np.expand_dims(py, 2)
        # self.coords = np.concatenate((xy, py), axis=2)


        self.global_sediment_grating = self.get_sediment()
        self.global_luminance_mask = self.get_luminance_mask()
        self.uv_scatter = self.get_local_scatter(self.max_uv_range, self.uv_light_decay_rate)
        self.red_scatter = self.get_local_scatter(self.max_red_range, self.red_light_decay_rate)

        self.prey_diameter = env_variables['prey_radius'] * 2
        self.prey_radius = env_variables['prey_radius']
        self.predator_size = env_variables['predator_radius'] * 2
        self.predator_radius = env_variables['predator_radius']


        self.FOV = FieldOfView(self.local_dim, self.max_red_range, self.arena_width, self.arena_height)

    def get_FOV_size(self):
        return self.local_dim, self.local_dim

    def get_sediment(self):

        if self.test_mode:
            # In test mode, use a fixed grating for testing purposes
            new_grating = np.zeros((self.arena_width, self.arena_height))
            # draw vertical stripes, 10 pixels in width
            for i in range(0, self.arena_width, 20):
                new_grating[:, i:i+10] = 1.0
            new_grating[self.arena_height // 2 - 80: self.arena_height // 2 - 30, self.arena_width // 2 + 30: self.arena_width // 2 + 80] = 0
            new_grating[self.arena_height // 2 + 40: self.arena_height // 2 + 120, self.arena_width // 2 + 20: self.arena_width // 2 + 100] = 10.

        else:
            new_grating = np.random.rand(self.arena_width, self.arena_height)
            new_grating = gaussian_filter(new_grating, sigma=self.sediment_sigma)
            new_grating -= np.min(new_grating)
            new_grating /= np.max(new_grating)

        return new_grating * self.bottom_intensity


    def get_local_scatter(self, max_range, decay_rate):
        """Computes effects of absorption and scatter, but incorporates effect of implicit scatter from line spread."""
        dim = max_range * 2 + 1
        x, y = np.arange(dim), np.arange(dim)
        y = np.expand_dims(y, 1)
        j = max_range + 1
        positional_mask = (((x - j) ** 2 + (y - j) ** 2) ** 0.5)  # Measure of distance from centre to every pixel
        desired_scatter = np.exp(-decay_rate * positional_mask)
        #desired_red_scatter = np.exp(-self.red_light_decay_rate * positional_mask)
        return desired_scatter


    def get_luminance_mask(self):
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


    def get_masked_bottom(self):

        # apply FOV portion of luminance mask
        local_luminance_mask = np.zeros((self.local_dim, self.local_dim))
        lum_slice = self.global_luminance_mask[self.FOV.enclosed_fov_top:self.FOV.enclosed_fov_bottom,
                                               self.FOV.enclosed_fov_left:self.FOV.enclosed_fov_right]
        local_luminance_mask[self.FOV.local_fov_top:self.FOV.local_fov_bottom,
                             self.FOV.local_fov_left:self.FOV.local_fov_right] = lum_slice

        return self.local_db * local_luminance_mask * self.red_scatter


    def reset(self):
        """To be called at start of episode"""
        self.global_sediment_grating = self.get_sediment()

    def erase(self):
        self.local_db = np.copy(self.base_db)

    def draw_sediment(self):
        """Slice the global sediment for current field of view"""

        sediment_slice = self.global_sediment_grating[self.FOV.enclosed_fov_top:self.FOV.enclosed_fov_bottom,
                                                      self.FOV.enclosed_fov_left:self.FOV.enclosed_fov_right]

        self.local_db[self.FOV.local_fov_top:self.FOV.local_fov_bottom,
                      self.FOV.local_fov_left:self.FOV.local_fov_right] = sediment_slice

