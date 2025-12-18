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

from simulation.eye import Eye

PHYS_DAMP = 0.7 ** 50
FISH_MOMENT_OF_INERTIA_MASS = 140
class Fish:


    def __init__(self, env_variables, max_uv_range, rng, actions):

        # For the purpose of producing a calibration curve.
        inertia = pymunk.moment_for_circle(FISH_MOMENT_OF_INERTIA_MASS, 0, env_variables['fish_head_radius'], (0, 0))
        self.max_uv_range = max_uv_range
        self.env_variables = env_variables
        self.body = pymunk.Body(1, inertia)
        self.rng = rng
        self.actions = actions
        self.num_actions = len(actions)
        # From mm, should be distance * pixels_per_mm * mass * (1-dampening^dt) / dt
        phys_dt = self.env_variables['sim_step_duration_seconds'] / self.env_variables['phys_steps_per_sim_step']
        self.distance_to_impulse_factor = 10 * 1 * (1 - PHYS_DAMP ** phys_dt) / phys_dt

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

        """Draw a bout angle and distance from the action distribution."""
        bout_vals = self.rng.multivariate_normal(action['mean'], action['cov'], 1)
        return bout_vals[0, 1], bout_vals[0, 0], action['mean'][1], action['mean'][0]

    def take_action(self, action_id):
        self.prev_action = action_id

        if self.env_variables['test_sensory_system']:
            self.body.angle += 0.1
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
        """Updates the current energy state for continuous and discrete fish."""
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
