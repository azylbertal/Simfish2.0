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

def read_elevation(fish_elevation, masked_pixels, eye_x, eye_y, elevation_angle, fish_angle, pr_angles, rf_size, predator_left, predator_right, predator_dist):


    eye_FOV_x = int(eye_x + (masked_pixels.shape[1] - 1) / 2)
    eye_FOV_y = int(eye_y + (masked_pixels.shape[0] - 1) / 2)
    radius = int(np.round(fish_elevation * np.tan(np.radians(elevation_angle))))
    circle_values, circle_angles = circle_edge_pixels(masked_pixels, (eye_FOV_y, eye_FOV_x), radius)
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
    l_ind = closest_index_parallel(circle_angles, bin_edges1)
    r_ind = closest_index_parallel(circle_angles, bin_edges2) + 1

    interleaved_edges = np.vstack((l_ind, r_ind)).T.flatten()
    bins_sum = np.add.reduceat(circle_values, interleaved_edges)[::2]
    counts = r_ind - l_ind
    pr_input = bins_sum / counts
    return pr_input

def read_prey_proj(max_uv_range, prey_diameter, eye_x, eye_y, uv_pr_angles, fish_angle, rf_size, lum_mask, prey_pos):
    """Reads the prey projection for the given eye position and fish angle.
    Same as " but performs more computation in parallel for each prey. Also have removed scatter.
    """
    ang_bin = 0.001  # this is the bin size for the projection
    proj_angles = np.arange(-np.pi, np.pi + ang_bin, ang_bin)  # this is the angle range for the projection

    eye_FOV_x = eye_x + (lum_mask.shape[1] - 1) / 2
    eye_FOV_y = eye_y + (lum_mask.shape[0] - 1) / 2
    rel_prey_pos = prey_pos - np.array([eye_FOV_x, eye_FOV_y])
    rho = np.hypot(rel_prey_pos[:, 0], rel_prey_pos[:, 1])

    within_range = np.where(rho < max_uv_range - 1)[0]
    prey_pos_in_range = prey_pos[within_range, :]
    rel_prey_pos = rel_prey_pos[within_range, :]
    rho = rho[within_range]
    theta = np.arctan2(rel_prey_pos[:, 1], rel_prey_pos[:, 0]) - fish_angle
    theta = np.arctan2(np.sin(theta),
                                                np.cos(theta))  # wrap to [-pi, pi]
    p_num = prey_pos_in_range.shape[0]

    half_angle = np.arctan(prey_diameter / (2 * rho))

    l_ind = closest_index_parallel(proj_angles, theta - half_angle).astype(int)
    r_ind = closest_index_parallel(proj_angles, theta + half_angle).astype(int)

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

    pr_ind_s = closest_index_parallel(proj_angles, uv_pr_angles - rf_size / 2)
    pr_ind_e = closest_index_parallel(proj_angles, uv_pr_angles + rf_size / 2)

    pr_occupation = (pr_ind_s[:, None] <= r) & (pr_ind_e[:, None] >= r)
    pr_occupation = pr_occupation.astype(float)
    pr_input = pr_occupation * np.expand_dims(total_angular_input, axis=0)
    pr_input = np.sum(pr_input, axis=1)

    return np.expand_dims(pr_input, axis=1)

def closest_index_parallel(array, value_array):
    """Find indices of the closest values in array (for each row in axis=0)."""
    value_array = np.expand_dims(value_array, axis=1)
    idxs = (np.abs(array - value_array)).argmin(axis=1)
    return idxs
