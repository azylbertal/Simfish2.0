# Copyright 2025 Asaph Zylbertal
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
from skimage.draw import circle_perimeter


class FieldOfView:

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
        """Computes effects of absorption and scatter, but incorporates effect of implicit scatter from line spread."""
        x, y = np.arange(self.local_dim), np.arange(self.local_dim)
        y = np.expand_dims(y, 1)
        j = self.max_visual_distance + 1
        positional_mask = (((x - j) ** 2 + (y - j) ** 2) ** 0.5)  # Measure of distance from centre to every pixel
        return np.exp(-self.light_decay_rate * positional_mask)

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

    def get_sliced_masked_image(self, img):

        # apply FOV portion of luminance mask
        masked_image = np.zeros((self.local_dim, self.local_dim))

        slice = img[self.enclosed_fov_top:self.enclosed_fov_bottom,
                    self.enclosed_fov_left:self.enclosed_fov_right]
        
        masked_image[self.local_fov_top:self.local_fov_bottom,
                             self.local_fov_left:self.local_fov_right] = slice


        return masked_image * self.local_scatter

def ray_sum(array, start_coord, angles_rad, step_size=0.5):
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

def circle_edge_pixels(image: np.ndarray, center: tuple, radius: int):
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