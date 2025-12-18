// geometry.cpp
// Copyright 2025 Asaph Zylbertal
// Licensed under the Apache License, Version 2.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

namespace py = pybind11;

// Helper function for closest index finding
std::vector<int> closest_index_parallel(const std::vector<double>& array, const std::vector<double>& value_array) {
    std::vector<int> indices(value_array.size());
    
    for (size_t i = 0; i < value_array.size(); ++i) {
        double target = value_array[i];
        double min_diff = std::numeric_limits<double>::infinity();
        int best_idx = 0;
        
        for (size_t j = 0; j < array.size(); ++j) {
            double diff = std::abs(array[j] - target);
            if (diff < min_diff) {
                min_diff = diff;
                best_idx = static_cast<int>(j);
            }
        }
        indices[i] = best_idx;
    }
    
    return indices;
}

// Ray sum function (optimized C++ version)
py::array_t<double> ray_sum_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> input,
    std::pair<double, double> start_coord,
    py::array_t<double> angles_input,
    double step_size = 0.5
) {
    auto buf = input.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Array must be 2D");
    }
    
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    double* array_ptr = static_cast<double*>(buf.ptr);
    
    auto angles_buf = angles_input.request();
    double* angles_ptr = static_cast<double*>(angles_buf.ptr);
    int n_angles = angles_buf.shape[0];
    
    double start_row = start_coord.first;
    double start_col = start_coord.second;
    
    if (start_row < 0 || start_row >= rows || start_col < 0 || start_col >= cols) {
        auto result = py::array_t<double>(n_angles);
        auto result_buf = result.request();
        std::fill_n(static_cast<double*>(result_buf.ptr), n_angles, 0.0);
        return result;
    }
    
    auto result = py::array_t<double>(n_angles);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    for (int angle_idx = 0; angle_idx < n_angles; ++angle_idx) {
        double angle = angles_ptr[angle_idx];
        double dy = std::sin(angle);
        double dx = std::cos(angle);
        
        double steps_to_right, steps_to_bottom;
        
        if (dx > 0) steps_to_right = (cols - 1 - start_col) / dx;
        else if (dx < 0) steps_to_right = -start_col / dx;
        else steps_to_right = std::numeric_limits<double>::infinity();
        
        if (dy > 0) steps_to_bottom = (rows - 1 - start_row) / dy;
        else if (dy < 0) steps_to_bottom = -start_row / dy;
        else steps_to_bottom = std::numeric_limits<double>::infinity();
        
        double max_steps = std::max(0.0, std::min(steps_to_right, steps_to_bottom));
        
        if (max_steps <= 0) {
            int idx = static_cast<int>(start_row) * cols + static_cast<int>(start_col);
            result_ptr[angle_idx] = array_ptr[idx];
            continue;
        }
        
        int num_steps = static_cast<int>(max_steps / step_size) + 1;
        double sum = 0.0;
        int prev_row = -1, prev_col = -1;
        
        for (int step = 0; step < num_steps; ++step) {
            double t = (step * max_steps) / std::max(1, num_steps - 1);
            if (t > max_steps) break;
            
            int row = static_cast<int>(std::round(start_row + t * dy));
            int col = static_cast<int>(std::round(start_col + t * dx));
            
            if (row >= 0 && row < rows && col >= 0 && col < cols &&
                (row != prev_row || col != prev_col)) {
                sum += array_ptr[row * cols + col];
                prev_row = row;
                prev_col = col;
            }
        }
        
        result_ptr[angle_idx] = sum;
    }
    
    return result;
}

// Circle edge pixels function
std::pair<py::array_t<double>, py::array_t<double>> circle_edge_pixels_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> image,
    std::pair<int, int> center,
    int radius
) {
    auto buf = image.request();
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    double* image_ptr = static_cast<double*>(buf.ptr);
    
    int center_row = center.first;
    int center_col = center.second;
    
    std::vector<double> pixels;
    std::vector<double> angles;
    
    // Bresenham circle algorithm for perimeter points
    int x = 0;
    int y = radius;
    int d = 3 - 2 * radius;
    
    auto add_octant_points = [&](int cx, int cy, int x, int y) {
        std::vector<std::pair<int, int>> points = {
            {cx + x, cy + y}, {cx - x, cy + y}, {cx + x, cy - y}, {cx - x, cy - y},
            {cx + y, cy + x}, {cx - y, cy + x}, {cx + y, cy - x}, {cx - y, cy - x}
        };
        
        for (const auto& point : points) {
            int row = point.first;
            int col = point.second;
            
            if (row >= 0 && row < rows && col >= 0 && col < cols) {
                pixels.push_back(image_ptr[row * cols + col]);
                angles.push_back(std::atan2(row - center_row, col - center_col));
            }
        }
    };
    
    add_octant_points(center_row, center_col, x, y);
    
    while (y >= x) {
        x++;
        if (d > 0) {
            y--;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
        add_octant_points(center_row, center_col, x, y);
    }
    
    // Sort by angles
    std::vector<size_t> indices(angles.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return angles[a] < angles[b];
    });
    
    std::vector<double> sorted_pixels(pixels.size());
    std::vector<double> sorted_angles(angles.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        sorted_pixels[i] = pixels[indices[i]];
        sorted_angles[i] = angles[indices[i]];
    }
    
    auto pixels_array = py::array_t<double>(sorted_pixels.size());
    auto angles_array = py::array_t<double>(sorted_angles.size());
    
    auto pixels_buf = pixels_array.request();
    auto angles_buf = angles_array.request();
    
    std::copy(sorted_pixels.begin(), sorted_pixels.end(), static_cast<double*>(pixels_buf.ptr));
    std::copy(sorted_angles.begin(), sorted_angles.end(), static_cast<double*>(angles_buf.ptr));
    
    return std::make_pair(pixels_array, angles_array);
}

// Read elevation function
py::array_t<double> read_elevation_cpp(
    double fish_elevation,
    py::array_t<double, py::array::c_style | py::array::forcecast> masked_pixels,
    double eye_x,
    double eye_y,
    double elevation_angle,
    double fish_angle,
    py::array_t<double> pr_angles,
    double rf_size,
    double predator_left,
    double predator_right,
    double predator_dist
) {
    auto masked_buf = masked_pixels.request();
    int rows = masked_buf.shape[0];
    int cols = masked_buf.shape[1];
    
    int eye_FOV_x = static_cast<int>(eye_x + (cols - 1) / 2);
    int eye_FOV_y = static_cast<int>(eye_y + (rows - 1) / 2);
    int radius = static_cast<int>(std::round(fish_elevation * std::tan(elevation_angle * M_PI / 180.0)));
    
    // Get circle edge pixels
    auto [circle_values_py, circle_angles_py] = circle_edge_pixels_cpp(masked_pixels, {eye_FOV_y, eye_FOV_x}, radius);
    
    auto circle_values_buf = circle_values_py.request();
    auto circle_angles_buf = circle_angles_py.request();
    
    int n_circle_points = circle_values_buf.shape[0];
    double* circle_values = static_cast<double*>(circle_values_buf.ptr);
    double* circle_angles = static_cast<double*>(circle_angles_buf.ptr);
    
    // Apply predator masking if applicable
    if (!std::isnan(predator_dist) && predator_dist < radius) {
        for (int i = 0; i < n_circle_points; ++i) {
            double angle = circle_angles[i];
            bool mask_condition;
            
            if (predator_left > predator_right) {
                mask_condition = (angle < predator_left) && (angle > predator_right);
            } else {
                mask_condition = (angle < predator_left) || (angle > predator_right);
            }
            
            if (mask_condition) {
                circle_values[i] = 0.0;
            }
        }
    }
    
    // Get PR angles
    auto pr_angles_buf = pr_angles.request();
    int n_pr_angles = pr_angles_buf.shape[0];
    double* pr_angles_ptr = static_cast<double*>(pr_angles_buf.ptr);
    
    // Correct PR angles
    std::vector<double> corrected_pr_angles(n_pr_angles);
    for (int i = 0; i < n_pr_angles; ++i) {
        double angle = pr_angles_ptr[i] + fish_angle;
        corrected_pr_angles[i] = std::atan2(std::sin(angle), std::cos(angle));
    }
    
    // Calculate bin edges
    std::vector<double> bin_edges1(n_pr_angles), bin_edges2(n_pr_angles);
    for (int i = 0; i < n_pr_angles; ++i) {
        bin_edges1[i] = corrected_pr_angles[i] - rf_size / 2;
        bin_edges2[i] = corrected_pr_angles[i] + rf_size / 2;
    }
    
    // Convert circle_angles to vector for closest_index_parallel
    std::vector<double> circle_angles_vec(circle_angles, circle_angles + n_circle_points);
    
    // Find closest indices
    auto l_ind = closest_index_parallel(circle_angles_vec, bin_edges1);
    auto r_ind = closest_index_parallel(circle_angles_vec, bin_edges2);
    
    // Calculate PR input
    auto result = py::array_t<double>(n_pr_angles);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    for (int i = 0; i < n_pr_angles; ++i) {
        int left_idx = l_ind[i];
        int right_idx = r_ind[i] + 1;
        
        // Ensure indices are within bounds
        left_idx = std::max(0, std::min(left_idx, n_circle_points - 1));
        right_idx = std::max(left_idx + 1, std::min(right_idx, n_circle_points));
        
        double sum = 0.0;
        int count = right_idx - left_idx;
        
        for (int j = left_idx; j < right_idx; ++j) {
            sum += circle_values[j];
        }
        
        result_ptr[i] = (count > 0) ? sum / count : 0.0;
    }
    
    return result;
}

// Read prey projection function
py::array_t<double> read_prey_proj_cpp(
    double max_uv_range,
    double prey_diameter,
    double eye_x,
    double eye_y,
    py::array_t<double> uv_pr_angles,
    double fish_angle,
    double rf_size,
    py::array_t<double, py::array::c_style | py::array::forcecast> lum_mask,
    py::array_t<double, py::array::c_style | py::array::forcecast> prey_pos
) {
    const double ang_bin = 0.001;
    
    // Create projection angles
    int n_proj_angles = static_cast<int>((2 * M_PI + ang_bin) / ang_bin);
    std::vector<double> proj_angles(n_proj_angles);
    for (int i = 0; i < n_proj_angles; ++i) {
        proj_angles[i] = -M_PI + i * ang_bin;
    }
    
    auto lum_buf = lum_mask.request();
    int rows = lum_buf.shape[0];
    int cols = lum_buf.shape[1];
    double* lum_ptr = static_cast<double*>(lum_buf.ptr);
    
    auto prey_buf = prey_pos.request();
    int n_prey = prey_buf.shape[0];
    double* prey_ptr = static_cast<double*>(prey_buf.ptr);
    
    double eye_FOV_x = eye_x + (cols - 1) / 2.0;
    double eye_FOV_y = eye_y + (rows - 1) / 2.0;
    
    // Filter prey within range
    std::vector<int> within_range_indices;
    std::vector<double> rho_values;
    std::vector<double> theta_values;
    std::vector<double> half_angles;
    std::vector<double> brightness_values;
    
    for (int i = 0; i < n_prey; ++i) {
        double prey_x = prey_ptr[i * 2];
        double prey_y = prey_ptr[i * 2 + 1];
        
        double rel_x = prey_x - eye_FOV_x;
        double rel_y = prey_y - eye_FOV_y;
        double rho = std::sqrt(rel_x * rel_x + rel_y * rel_y);
        
        if (rho < max_uv_range - 1) {
            within_range_indices.push_back(i);
            rho_values.push_back(rho);
            
            double theta = std::atan2(rel_y, rel_x) - fish_angle;
            theta = std::atan2(std::sin(theta), std::cos(theta)); // wrap to [-pi, pi]
            theta_values.push_back(theta);
            
            half_angles.push_back(std::atan(prey_diameter / (2 * rho)));
            
            // Get brightness (with bounds checking)
            int y_idx = static_cast<int>(std::floor(prey_y)) - 1;
            int x_idx = static_cast<int>(std::floor(prey_x)) - 1;
            y_idx = std::max(0, std::min(y_idx, rows - 1));
            x_idx = std::max(0, std::min(x_idx, cols - 1));
            brightness_values.push_back(lum_ptr[y_idx * cols + x_idx]);
        }
    }
    
    int p_num = within_range_indices.size();
    
    // Calculate total angular input
    std::vector<double> total_angular_input(n_proj_angles, 0.0);
    
    for (int i = 0; i < p_num; ++i) {
        double theta = theta_values[i];
        double half_angle = half_angles[i];
        double brightness = brightness_values[i];
        
        std::vector<double> left_angles = {theta - half_angle};
        std::vector<double> right_angles = {theta + half_angle};
        
        auto l_ind = closest_index_parallel(proj_angles, left_angles);
        auto r_ind = closest_index_parallel(proj_angles, right_angles);
        
        for (int j = l_ind[0]; j <= r_ind[0] && j < n_proj_angles; ++j) {
            total_angular_input[j] += brightness;
        }
    }
    
    // Calculate PR input
    auto uv_pr_buf = uv_pr_angles.request();
    int n_uv_pr = uv_pr_buf.shape[0];
    double* uv_pr_ptr = static_cast<double*>(uv_pr_buf.ptr);
    
    auto result = py::array_t<double>({n_uv_pr, 1});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    for (int i = 0; i < n_uv_pr; ++i) {
        double angle = uv_pr_ptr[i];
        std::vector<double> start_angles = {angle - rf_size / 2};
        std::vector<double> end_angles = {angle + rf_size / 2};
        
        auto pr_ind_s = closest_index_parallel(proj_angles, start_angles);
        auto pr_ind_e = closest_index_parallel(proj_angles, end_angles);
        
        double sum = 0.0;
        for (int j = pr_ind_s[0]; j <= pr_ind_e[0] && j < n_proj_angles; ++j) {
            sum += total_angular_input[j];
        }
        
        result_ptr[i] = sum;
    }
    
    return result;
}


// Pybind11 module definition
PYBIND11_MODULE(geometry, m) {
    m.doc() = "Fast C++ implementation of geometry functions";
    
    m.def("ray_sum", &ray_sum_cpp, 
          "Compute sum of pixel values along rays",
          py::arg("array"), 
          py::arg("start_coord"), 
          py::arg("angles_rad"), 
          py::arg("step_size") = 0.5);
    
    m.def("circle_edge_pixels", &circle_edge_pixels_cpp,
          "Extract pixel values from the edge of a circular region",
          py::arg("image"),
          py::arg("center"),
          py::arg("radius"));
    
    m.def("read_elevation", &read_elevation_cpp,
          "Read elevation data for fish visual system",
          py::arg("fish_elevation"),
          py::arg("masked_pixels"),
          py::arg("eye_x"),
          py::arg("eye_y"),
          py::arg("elevation_angle"),
          py::arg("fish_angle"),
          py::arg("pr_angles"),
          py::arg("rf_size"),
          py::arg("predator_left"),
          py::arg("predator_right"),
          py::arg("predator_dist"));
    
    m.def("read_prey_proj", &read_prey_proj_cpp,
          "Read prey projection for UV photoreceptors",
          py::arg("max_uv_range"),
          py::arg("prey_diameter"),
          py::arg("eye_x"),
          py::arg("eye_y"),
          py::arg("uv_pr_angles"),
          py::arg("fish_angle"),
          py::arg("rf_size"),
          py::arg("lum_mask"),
          py::arg("prey_pos"));
    
}