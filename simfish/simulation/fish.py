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
import pymunk
from .constants import MEDIUM_MASS, PHYS_DAMP, PIXELS_PER_MM
from .eye import Eye

class Fish:
    """
    A class representing a fish agent in a 2D physics-based simulation environment.
    The Fish class models a fish with physical properties (body, head, mouth, tail), visual
    perception through two eyes, energy management, and action execution capabilities. It uses
    the pymunk physics engine for realistic movement and collision detection.
    Attributes:
        max_uv_range (float): Maximum range for UV light perception.
        env_variables (dict): Dictionary containing environment configuration parameters.
        body (pymunk.Body): The physics body representing the fish.
        rng (numpy.random.Generator): Random number generator for stochastic actions.
        actions (list): List of available actions the fish can perform.
        num_actions (int): Number of available actions.
        distance_to_impulse_factor (float): Conversion factor from distance to physics impulse.
        mouth (pymunk.Circle): Circular collision shape representing the fish's mouth.
        head (pymunk.Circle): Circular collision shape representing the fish's head.
        tail (pymunk.Poly): Polygonal collision shape representing the fish's tail.
        verg_angle (float): Vergence angle between the two eyes in radians.
        retinal_field (float): Visual field angle for each eye in radians.
        conv_state (int): Current convergence state of the eyes.
        left_eye (Eye): Left eye object for visual perception.
        right_eye (Eye): Right eye object for visual perception.
        hungry (float): Hunger level of the fish.
        prey_consumed (bool): Whether prey was consumed in the current step.
        touched_edge (bool): Whether the fish has touched the environment edge.
        touched_predator (bool): Whether the fish has touched a predator.
        making_capture (bool): Whether the fish is currently performing a capture action.
        capture_possible (bool): Whether a capture is currently possible.
        prev_action_distance (float): Distance component of the previous action.
        prev_action_angle (float): Angle component of the previous action.
        prev_action (int): Index of the previously executed action.
        energy_level (float): Current energy level of the fish (0.0 to 1.0).
        d_scaling_energy_cost (float): Energy cost scaling factor for distance-based movement.
        a_scaling_energy_cost (float): Energy cost scaling factor for angular movement.
        baseline_energy_use (float): Baseline energy consumption per time step.
        action_energy_reward_scaling (float): Scaling factor for energy-based rewards.
        touched_edge_this_step (bool): Whether the fish touched the edge in the current step.
        deterministic_action (bool): Whether to use deterministic (mean) action values.
        - The fish uses a physics simulation with configurable mass, inertia, and damping.
        - Collision types are set for different body parts to enable specific interactions.
        - Energy management includes consumption tracking and rewards based on energy efficiency.
        - The fish can operate in test modes (sensory system testing, paralysis) for debugging.

    """
    def __init__(self, env_variables, max_uv_range, rng, actions):

        # For the purpose of producing a calibration curve.
        fish_inertia = pymunk.moment_for_circle(MEDIUM_MASS, 0, env_variables['fish_head_radius'], (0, 0))
        self.max_uv_range = max_uv_range
        self.env_variables = env_variables
        self.body = pymunk.Body(MEDIUM_MASS, fish_inertia)
        self.rng = rng
        self.actions = actions
        self.num_actions = len(actions)
        # From mm, should be distance * pixels_per_mm * mass * (1-dampening^dt) / dt
        phys_dt = self.env_variables['sim_step_duration_seconds'] / self.env_variables['phys_steps_per_sim_step']
        self.distance_to_impulse_factor = PIXELS_PER_MM * MEDIUM_MASS * (1 - PHYS_DAMP ** phys_dt) / phys_dt

        # Mouth
        self.mouth = pymunk.Circle(self.body, env_variables['fish_mouth_radius'], offset=(0, 0))
        self.mouth.elasticity = 1.0
        self.mouth.collision_type = 3

        # Head
        self.head = pymunk.Circle(self.body, env_variables['fish_head_radius'],
                                  offset=(-env_variables['fish_head_radius'], 0))
        self.head.elasticity = 1.0
        self.head.collision_type = 6

        # # Tail
        tail_coordinates = ((-env_variables['fish_head_radius'], 0),
                            (-env_variables['fish_head_radius'], - env_variables['fish_head_radius']),
                            (-env_variables['fish_head_radius'] - env_variables['fish_tail_length'], 0),
                            (-env_variables['fish_head_radius'], env_variables['fish_head_radius']))
        self.tail = pymunk.Poly(self.body, tail_coordinates)
        self.tail.elasticity = 1.0
        self.tail.collision_type = 6

        # Init visual fields.
        self.verg_angle = env_variables['eyes_verg_angle'] * (np.pi / 180)
        self.retinal_field = env_variables['eyes_visual_field'] * (np.pi / 180)
        self.conv_state = 0


        self.left_eye = Eye(self.verg_angle, self.retinal_field, True, env_variables,
                            max_uv_range=self.max_uv_range, rng=self.rng)
        self.right_eye = Eye(self.verg_angle, self.retinal_field, False, env_variables,
                             max_uv_range=self.max_uv_range, rng=self.rng)

        self.hungry = 0
        self.prey_consumed = False
        self.touched_edge = False
        self.touched_predator = False
        self.making_capture = False
        self.capture_possible = False
        self.prev_action_distance = 0
        self.prev_action_angle = 0
        self.prev_action = 0

        # Energy system (new simulation)
        self.energy_level = 1.0
        self.d_scaling_energy_cost = self.env_variables['energy_distance_factor']
        self.a_scaling_energy_cost = self.env_variables['energy_angle_factor']
        self.baseline_energy_use = self.env_variables['energy_baseline']

        self.action_energy_reward_scaling = self.env_variables['reward_energy_use_factor']

        # Touch edge - for penalty
        self.touched_edge_this_step = False

        self.deterministic_action = self.env_variables['fish_deterministic_action']

    def draw_angle_dist(self, action):
        """
        Draw angle and distance values from a multivariate normal distribution.
        This method samples from a multivariate normal distribution defined by the given
        action parameters and returns both the sampled values and the mean values.
        Args:
            action (dict): A dictionary containing:
                - 'mean': array-like of shape (2,) representing the mean values [distance_mean, angle_mean]
                - 'cov': array-like of shape (2, 2) representing the covariance matrix
        Returns:
            tuple: A 4-tuple containing:
                - float: Sampled angle value
                - float: Sampled distance value
                - float: Mean angle value
                - float: Mean distance value
        """

        bout_vals = self.rng.multivariate_normal(action['mean'], action['cov'], 1)
        return bout_vals[0, 1], bout_vals[0, 0], action['mean'][1], action['mean'][0]

    def take_action(self, action_id):
        """
        Execute a specific action for the fish agent in the simulation.
        This method updates the fish's position and orientation based on the selected action.
        It handles different simulation modes including sensory system testing and fish paralysis.
        Args:
            action_id (int): The identifier of the action to be executed. Must be within
                            the range of available actions (0 to num_actions-1).
        Side Effects:
            - Updates self.prev_action with the current action_id
            - Updates self.prev_action_distance and self.prev_action_angle
            - Modifies self.body.angle based on the action or test mode sequence
            - Applies impulse to self.body if fish is not paralyzed
            - Sets self.making_capture to True if action is a capture action
        Behavior Modes:
            - test_sensory_system: Rotates fish body by 0.1 radians, no movement
            - fish_paralyze: Sets distance and angle changes to 0
            - normal: Draws angle and distance from action distribution, applies impulse
        Notes:
            - If deterministic_action is True, uses mean values instead of sampling
            - The impulse applied is scaled by distance_to_impulse_factor
        """
        self.prev_action = action_id

        if self.env_variables['test_sensory_system']:
            self.body.angle += 0.1
            self.prev_action_distance = 0
            self.prev_action_angle = 0

        else:
            if "fish_paralyze" in self.env_variables and self.env_variables["fish_paralyze"]:
                
                self.prev_action_distance = 0
                self.prev_action_angle = 0
            else:
                angle_change, distance, mean_angle, mean_distance = self.draw_angle_dist(self.actions[action_id])
                if self.deterministic_action:
                    angle_change = mean_angle
                    distance = mean_distance

                if self.actions[action_id]['is_capture']:
                    self.making_capture = True

                self.prev_action_angle = angle_change
                self.body.angle += self.prev_action_angle
                self.prev_action_distance = distance
                self.body.apply_impulse_at_local_point((self.distance_to_impulse_factor * self.prev_action_distance, 0))

        if not action_id in range(self.num_actions):
            print("Invalid action given")

    def update_energy_level(self, consumption):
        """
        Update the fish's energy level based on consumption and energy costs.
        This method calculates the net energy change by considering energy gained from
        consumption and energy expended through movement and actions. It also computes
        a reward based on energy expenditure.
        Args:
            consumption (float): Amount of energy gained from consuming food or prey.
        Returns:
            float: Reward value calculated as negative energy use scaled by 
                   action_energy_reward_scaling.
        Side Effects:
            - Updates self.energy_level based on net energy change (unless fixed energy
              level mode is enabled via env_variables)
            - Caps energy_level at 1.0 maximum

        Notes:
            - If env_variables["fish_fixed_energy_level"] is True, energy level 
              remains unchanged
            - Energy level is capped at 1.0 after update
        """
        energy_gain = consumption

        energy_use = self.d_scaling_energy_cost * (abs(self.prev_action_distance)) + \
                        self.a_scaling_energy_cost * (abs(self.prev_action_angle)) + \
                        self.baseline_energy_use
            
        if self.actions[self.prev_action]['is_capture']:
            energy_use *= self.env_variables['capture_swim_energy_cost_scaling']
        
        energy_change = energy_gain - energy_use

        reward = -energy_use * self.action_energy_reward_scaling

        if "fish_fixed_energy_level" in self.env_variables and self.env_variables["fish_fixed_energy_level"]:
            pass
        else:
            self.energy_level += energy_change
            self.energy_level = min(self.energy_level, 1.0)  # Ensure energy level does not exceed 1.0
        return reward
