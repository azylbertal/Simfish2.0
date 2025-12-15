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
import dm_env
from dm_env import specs
import time

from simulation.arena import Arena
from simulation.fish import Fish
from acme.wrappers import observation_action_reward
OAR = observation_action_reward.OAR


class BaseEnvironment(dm_env.Environment):
    """Base class for the Simfish environment."""

    def __init__(self, env_variables, actions, seed=None):

        super().__init__()
        self.rng = np.random.default_rng(seed=seed)
        self.env_variables = env_variables
        self.actions = actions
        self.num_actions = len(actions)
        
        self.max_uv_range = np.absolute(np.log(0.001) / self.env_variables["arena_light_decay_rate"])

        self.arena = Arena(self.env_variables, rng=self.rng)
        self.dark_row = int(self.env_variables['arena_height'] * self.env_variables['arena_dark_fraction'])

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = self.env_variables['phys_drag']

        self.fish = Fish(env_variables=env_variables, max_uv_range=self.max_uv_range, rng=self.rng, actions=actions)       

        if self.env_variables["salt"]:
            self.salt_gradient = None
            self.xp, self.yp = np.arange(self.env_variables['arena_width']), np.arange(
                self.env_variables['arena_height'])
        self.salt_location = None

        self.capture_fraction = int(
            self.env_variables["phys_steps_per_sim_step"] * self.env_variables['capture_swim_permisive_time_fraction'])
        self.capture_start = 1  # int((self.env_variables['phys_steps_per_sim_step'] - self.capture_fraction) / 2)
        self.capture_end = self.capture_start + self.capture_fraction


        self.space.add(self.fish.body, self.fish.mouth, self.fish.head, self.fish.tail)
        self.prey_shapes = []
        self.prey_cloud_wall_shapes = []
        self.predator_shape = None
        self.energy_associated_reward = 0
        self.consumption_associated_reward = 0
        self.salt_associated_reward = 0
        self.predator_associated_reward = 0
        self.wall_associated_reward = 0
        self._prey_escape_p_per_physics_step = 1 - (1-self.env_variables["prey_p_escape"])**(1/self.env_variables["phys_steps_per_sim_step"])

        self.create_walls()

        self.set_collisions()

        self.continuous_actions = False
        self._reset_next_step = True
        self.action_used = np.zeros(self.num_actions)



    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.tested_predator = False
        self.num_steps = 0
        self.fish.touched_edge_this_step = False
        self.prey_caught = 0
        self.predator_attacks_avoided = 0
        self.energy_level_log = []
        self.salt_concentration = 0
        self.switch_step = None

        if "fish_init_energy_level" in self.env_variables:
            self.fish.energy_level = self.env_variables["fish_init_energy_level"]
        else:
            self.fish.energy_level = 1

        # Reset salt gradient
        if self.env_variables["salt"]:
            self.reset_salt_gradient()
        else:
            self.salt_location = [np.nan, np.nan]

        self.clear_environmental_features()
        self.arena.reset()

        self.mask_buffer = []
        self.action_buffer = []
        self.position_buffer = []
        self.fish_angle_buffer = []

        self.failed_capture_attempts = 0
        if self.env_variables['test_sensory_system']:
            self.fish.body.position = (self.env_variables['arena_width'] / 2, self.env_variables['arena_height'] / 2)
            self.fish.body.angle = 0
        else:
            self.fish.body.position = (self.rng.integers(self.env_variables['fish_mouth_radius'] + 40,
                                                        self.env_variables['arena_width'] - (self.env_variables[
                                                                                                'fish_mouth_radius'] + 40)),
                                    self.rng.integers(self.env_variables['fish_mouth_radius'] + 40,
                                                        self.env_variables['arena_height'] - (self.env_variables[
                                                                                                'fish_mouth_radius'] + 40)))
            self.fish.body.angle = self.rng.random() * 2 * np.pi

        self.fish.body.velocity = (0, 0)
        self.fish.capture_possible = False

        if self.env_variables["prey_cloud_num"] > 0:
            self.prey_cloud_locations = [
                [self.rng.integers(
                    low=(self.env_variables["prey_cloud_region_size"] / 2) + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                    high=self.env_variables['arena_width'] - (
                            self.env_variables['prey_radius'] + self.env_variables[
                        'fish_mouth_radius']) - (self.env_variables["prey_cloud_region_size"] / 2)),
                 self.rng.integers(
                     low=(self.env_variables["prey_cloud_region_size"] / 2) + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                     high=self.env_variables['arena_height'] - (
                             self.env_variables['prey_radius'] + self.env_variables[
                         'fish_mouth_radius']) - (self.env_variables["prey_cloud_region_size"] / 2))]
                for cloud in range(int(self.env_variables["prey_cloud_num"]))]

            if not self.env_variables["prey_reproduction_mode"]:
                self.build_prey_cloud_walls()
        if self.env_variables["test_sensory_system"]:
            self.create_prey(prey_position=(self.env_variables['arena_width'] / 2 + 30,
                                            self.env_variables['arena_height'] / 2 + 30))
            self.create_prey(prey_position=(self.env_variables['arena_width'] / 2,
                                            self.env_variables['arena_height'] / 2 - 40))
            self.create_prey(prey_position=(self.env_variables['arena_width'] / 2,
                                            self.env_variables['arena_height'] / 2 + 40))
            self.create_prey(prey_position=(self.env_variables['arena_width'] / 2 + 30,
                                            self.env_variables['arena_height'] / 2 - 30))
            self.create_prey(prey_position=(self.env_variables['arena_width'] / 2 + 5,
                                            self.env_variables['arena_height'] / 2 + 50))
            
        else:
            for i in range(int(self.env_variables['prey_num'])):
                self.create_prey()

        self.recent_cause_of_death = None
        self.available_prey = self.env_variables["prey_num"]
        self.vector_agreement = []

        self.predator_body = None
        self.predator_shape = None
        self.predator_target = None

        self.last_action = None
        self.prey_consumed_this_step = False
        self.event_captured_by_predator = False
        self.event_survived_predator = False

        self.survived_attack = False
        self.predator_prob = np.zeros(self.env_variables['max_episode_length'])

        predator_epoch_starts = self.rng.integers(
            low=self.env_variables['predator_immunity_steps'], high=self.env_variables['max_episode_length'] - self.env_variables['predator_epoch_duration'],
            size=self.env_variables['predator_epoch_num'])
        for i in predator_epoch_starts:
            self.predator_prob[i:i + self.env_variables['predator_epoch_duration']] = self.env_variables['predator_probability_per_epoch_step']

        # For Reward tracking (debugging)
        print(f"""REWARD CONTRIBUTIONS:        
              Energy: {self.energy_associated_reward}
              Consumption: {self.consumption_associated_reward}
              Salt: {self.salt_associated_reward}
              Predator: {self.predator_associated_reward}
              Wall: {self.wall_associated_reward}
              """)
        print(f"actions used: {self.action_used / np.sum(self.action_used)}")
        self.energy_associated_reward = 0
        self.consumption_associated_reward = 0
        self.salt_associated_reward = 0
        self.predator_associated_reward = 0
        self.wall_associated_reward = 0

        self.total_attacks_avoided = 0
        self.total_attacks_captured = 0
        self.action_used = np.zeros(self.num_actions)

        return dm_env.restart(self.get_observation(action=0, reward=0.))

    def set_collisions(self):
        """Specifies the collisions that occur in the Pymunk simulation."""

        # Collision Types:
        # 1: Edge
        # 2: Prey
        # 3: Fish mouth
        # 4: Sand grains (not implemented yet)
        # 5: Predator
        # 6: Fish body
        # 7: Prey cloud wall

        self.prey_col = self.space.add_collision_handler(2, 3)
        self.prey_col.begin = self.prey_touch_mouth
        self.prey_col2 = self.space.add_collision_handler(2, 6)
        self.prey_col2.begin = self.prey_touch_body


        self.pred_col = self.space.add_collision_handler(5, 3)
        self.pred_col.begin = self.touch_predator
        self.pred_col2 = self.space.add_collision_handler(5, 6)
        self.pred_col2.begin = self.touch_predator

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_wall

        self.edge_pred_col = self.space.add_collision_handler(1, 5)
        self.edge_pred_col.begin = self.remove_predator

        self.prey_pred_col = self.space.add_collision_handler(2, 5)
        self.prey_pred_col.begin = self.no_collision

        # To prevent the differential wall being hit by fish
        self.fish_prey_wall = self.space.add_collision_handler(3, 7)
        self.fish_prey_wall.begin = self.no_collision
        self.fish_prey_wall2 = self.space.add_collision_handler(6, 7)
        self.fish_prey_wall2.begin = self.no_collision
        self.pred_prey_wall2 = self.space.add_collision_handler(5, 7)
        self.pred_prey_wall2.begin = self.no_collision

    def clear_environmental_features(self):
        """Removes all prey and predators from the simulation"""
        for i, shp in enumerate(self.prey_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.prey_cloud_wall_shapes):
            self.space.remove(shp)

        self.prey_cloud_wall_shapes = []

        if self.predator_shape is not None:
            self.remove_predator()
        self.predator_location = None

        self.prey_shapes = []
        self.prey_bodies = []
        self.prey_identifiers = []
        self.paramecia_gaits = []
        if self.env_variables["prey_reproduction_mode"]:
            self.prey_ages = []
        self.total_prey_created = 0

    def reproduce_prey(self):
        num_prey = len(self.prey_bodies)
        p_prey_birth = self.env_variables["prey_birth_rate"] * (self.env_variables["prey_num"] - num_prey)
        for cloud in self.prey_cloud_locations:
            if self.rng.random(1) < p_prey_birth:
                if not self.check_proximity(cloud, self.env_variables["prey_cloud_region_size"]):
                    new_location = (
                        self.rng.integers(low=cloud[0] - (self.env_variables["prey_cloud_region_size"] / 2),
                                          high=cloud[0] + (self.env_variables["prey_cloud_region_size"] / 2)),
                        self.rng.integers(low=cloud[1] - (self.env_variables["prey_cloud_region_size"] / 2),
                                          high=cloud[1] + (self.env_variables["prey_cloud_region_size"] / 2))
                    )
                    self.create_prey(new_location)
                    self.available_prey += 1

    def reset_salt_gradient(self, salt_source=None):
        if salt_source is None:
            salt_source_x = self.rng.integers(0, self.env_variables['arena_width'] - 1)
            salt_source_y = self.rng.integers(0, self.env_variables['arena_height'] - 1)
        else:
            salt_source_x = salt_source[0]
            salt_source_y = salt_source[1]

        self.salt_location = [salt_source_x, salt_source_y]
        salt_distance = (((salt_source_x - self.xp[:, None]) ** 2 + (
                salt_source_y - self.yp[None, :]) ** 2) ** 0.5)  # Measure of distance from source at every point.
        self.salt_gradient = np.exp(-self.env_variables["salt_concentration_decay"] * salt_distance) 

    def build_prey_cloud_walls(self):
        for i in self.prey_cloud_locations:
            half_cloud_size = (self.env_variables["prey_cloud_region_size"] / 2)
            wall_edges = [
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - half_cloud_size, i[1] - half_cloud_size), (i[0] - half_cloud_size, i[1] + half_cloud_size), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - half_cloud_size, i[1] + half_cloud_size), (i[0] + half_cloud_size, i[1] + half_cloud_size), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] + half_cloud_size, i[1] + half_cloud_size), (i[0] + half_cloud_size, i[1] - half_cloud_size), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - half_cloud_size, i[1] - half_cloud_size), (i[0] + half_cloud_size, i[1] - half_cloud_size), 1)
            ]
            for s in wall_edges:
                s.friction = 1.
                s.group = 1
                s.collision_type = 7
                self.space.add(s)
                self.prey_cloud_wall_shapes.append(s)

    def create_walls(self):
        wall_width = 5
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, wall_width), (0, self.env_variables['arena_height']), wall_width),
            pymunk.Segment(
                self.space.static_body,
                (wall_width, self.env_variables['arena_height']),
                (self.env_variables['arena_width'], self.env_variables['arena_height']),
                wall_width),
            pymunk.Segment(
                self.space.static_body,
                (self.env_variables['arena_width'] - wall_width, self.env_variables['arena_height']),
                (self.env_variables['arena_width'] - wall_width, wall_width),
                wall_width),
            pymunk.Segment(
                self.space.static_body,
                (wall_width, wall_width), (self.env_variables['arena_width'], wall_width), wall_width)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            self.space.add(s)

    @staticmethod
    def no_collision(arbiter, space, data):
        return False

    def touch_wall(self, arbiter, space, data):
        if not self.env_variables["arena_wall_bounce"]:
            return self._touch_wall(arbiter, space, data)
        else:
            return self._touch_wall_reflect(arbiter, space, data)

    def _touch_wall_reflect(self, arbiter, space, data):
        new_position_x = self.fish.body.position[0]
        new_position_y = self.fish.body.position[1]

        if new_position_x < 40:  # Wall d
            new_position_x = 40 + self.env_variables["fish_head_radius"] + \
                             self.env_variables["fish_tail_length"]
        elif new_position_x > self.env_variables['arena_width'] - 40:  # wall b
            new_position_x = self.env_variables['arena_width'] - (
                    40 + self.env_variables["fish_head_radius"] +
                    self.env_variables["fish_tail_length"])
        if new_position_y < 40:  # wall a
            new_position_y = 40 + self.env_variables["fish_head_radius"] + \
                             self.env_variables["fish_tail_length"]
        elif new_position_y > self.env_variables['arena_height'] - 40:  # wall c
            new_position_y = self.env_variables['arena_height'] - (
                    40 + self.env_variables["fish_head_radius"] +
                    self.env_variables["fish_tail_length"])

        new_position = pymunk.Vec2d(new_position_x, new_position_y)
        self.fish.body.position = new_position
        self.fish.body.velocity = (0, 0)

        if self.fish.body.angle < np.pi:
            self.fish.body.angle += np.pi
        else:
            self.fish.body.angle -= np.pi
        self.fish.touched_edge = True
        return True

    def _touch_wall(self, arbiter, space, data):
        position_x = self.fish.body.position[0]
        position_y = self.fish.body.position[1]
        new_position_x = position_x
        new_position_y = position_y

        if position_x < 8:
            new_position_x = 10
        elif position_x > self.env_variables["arena_width"] - 7:
            new_position_x = self.env_variables["arena_width"] - 9

        if position_y < 8:
            new_position_y = 10
        elif position_y > self.env_variables["arena_height"] - 7:
            new_position_y = self.env_variables["arena_height"] - 9

        if "new_position_x" in locals():
            new_position = pymunk.Vec2d(new_position_x, self.fish.body.position[1])
            self.fish.body.position = new_position
            self.fish.body.velocity = (0, 0)

        if "new_position_y" in locals():
            new_position = pymunk.Vec2d(self.fish.body.position[0], new_position_y)
            self.fish.body.position = new_position
            self.fish.body.velocity = (0, 0)

        self.fish.touched_edge = True
        self.fish.touched_edge_this_step = True
        return True

    def create_prey(self, prey_position=None, prey_orientation=None, prey_gait=None, prey_age=None):
        self.prey_bodies.append(pymunk.Body(self.env_variables['prey_mass'], self.env_variables['prey_inertia']))
        self.prey_shapes.append(pymunk.Circle(self.prey_bodies[-1], self.env_variables['prey_radius']))
        self.prey_shapes[-1].elasticity = 1.0
        self.prey_bodies[-1].angle = self.rng.uniform(0, np.pi * 2)
        if prey_position is None:
            if self.env_variables["prey_cloud_num"] == 0:
                self.prey_bodies[-1].position = (
                    self.rng.integers(
                        self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'] + 40,
                        self.env_variables['arena_width'] - (
                                self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'] +
                                40)),
                    self.rng.integers(
                        self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'] + 40,
                        self.env_variables['arena_height'] - (
                                self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'] +
                                40)))
            else:
                cloud = self.rng.choice(self.prey_cloud_locations)
                self.prey_bodies[-1].position = (
                    self.rng.integers(low=cloud[0] - (self.env_variables["prey_cloud_region_size"] / 2),
                                      high=cloud[0] + (self.env_variables["prey_cloud_region_size"] / 2)),
                    self.rng.integers(low=cloud[1] - (self.env_variables["prey_cloud_region_size"] / 2),
                                      high=cloud[1] + (self.env_variables["prey_cloud_region_size"] / 2))
                )
        else:
            self.prey_bodies[-1].position = prey_position
            if prey_orientation is not None:
                self.prey_bodies[-1].angle = prey_orientation

        if prey_orientation is None:
            # When is a new prey being created
            self.prey_identifiers.append(self.total_prey_created)
            self.total_prey_created += 1
            self.paramecia_gaits.append(
                self.rng.choice([0, 1, 2], 1, p=[1 - (self.env_variables["prey_p_fast"] + self.env_variables["prey_p_slow"]),
                                                  self.env_variables["prey_p_slow"],
                                                  self.env_variables["prey_p_fast"]])[0])
            if self.env_variables["prey_reproduction_mode"]:
                self.prey_ages.append(0)
        else:

            self.paramecia_gaits.append(int(prey_gait))
            if self.env_variables["prey_reproduction_mode"]:
                self.prey_ages.append(int(prey_age))

        self.prey_shapes[-1].collision_type = 2
        self.space.add(self.prey_bodies[-1], self.prey_shapes[-1])

    def check_proximity(self, feature_position, sensing_distance):
        sensing_area = [[feature_position[0] - sensing_distance,
                         feature_position[0] + sensing_distance],
                        [feature_position[1] - sensing_distance,
                         feature_position[1] + sensing_distance]]
        is_in_area = sensing_area[0][0] <= self.fish.body.position[0] <= sensing_area[0][1] and \
                     sensing_area[1][0] <= self.fish.body.position[1] <= sensing_area[1][1]
        if is_in_area:
            return True
        else:
            return False

    def check_proximity_all_prey(self, sensing_distance):
        all_prey_positions = np.array([pr.position for pr in self.prey_bodies])

        fish_position = np.expand_dims(np.array(self.fish.body.position), 0)
        fish_prey_vectors = all_prey_positions - fish_position

        fish_prey_distances = ((fish_prey_vectors[:, 0] ** 2) + (fish_prey_vectors[:, 1] ** 2)) ** 0.5
        within_range = fish_prey_distances < sensing_distance
        return within_range

    def move_prey(self, micro_step):
        if len(self.prey_bodies) == 0:
            return

        # Generate impulses
        impulse_types = [0, self.env_variables["prey_impulse_slow"], self.env_variables["prey_impulse_fast"]]
        impulses = [impulse_types[gait] for gait in self.paramecia_gaits]
        if not self.fish.prey_consumed:
            for touched_index in self.touched_prey_indices: # Impulse from being touched by fish
                impulses[touched_index] += self.env_variables["prey_impulse_jump"]            

        # Do once per step.
        if micro_step == 0:
            gaits_to_switch = self.rng.random(len(self.prey_shapes)) < self.env_variables["prey_p_switch"]
            switch_to = self.rng.choice([0, 1, 2], len(self.prey_shapes),
                                         p=[1 - (self.env_variables["prey_p_slow"] + self.env_variables["prey_p_fast"]),
                                            self.env_variables["prey_p_slow"], self.env_variables["prey_p_fast"]])
            self.paramecia_gaits = [switch_to[i] if gaits_to_switch[i] else old_gait for i, old_gait in
                                    enumerate(self.paramecia_gaits)]

            # Angles of change
            angle_changes = self.rng.uniform(-self.env_variables['prey_max_turning_angle'],
                                              self.env_variables['prey_max_turning_angle'],
                                              len(self.prey_shapes))

            # Large angle changes
            large_turns = self.rng.uniform(-np.pi, np.pi, len(self.prey_shapes))
            large_turns_implemented = self.rng.random(len(self.prey_shapes)) < self.env_variables["prey_p_large_turn"]
            angle_changes = angle_changes + (large_turns * large_turns_implemented)

            self.prey_within_range = self.check_proximity_all_prey(self.env_variables["prey_sensing_distance"])

        for i, prey_body in enumerate(self.prey_bodies):
            if micro_step == 0:
                prey_body.angle = prey_body.angle + angle_changes[i]
            prey_body.apply_impulse_at_local_point((impulses[i], 0))

            if self.prey_within_range[i]:
                # Motion from prey escape
                if self.rng.random() < self._prey_escape_p_per_physics_step:
                    prey_body.apply_impulse_at_local_point((self.env_variables["prey_impulse_jump"], 0))



    def prey_touch_mouth(self, arbiter, space, data):
        valid_capture = False
        for i, shp in enumerate(self.prey_shapes):
            if shp == arbiter.shapes[0]:
                touched_prey_index = i
                break
                
        if self.fish.capture_possible:
            # Check if angles line up.
            prey_position = self.prey_bodies[touched_prey_index].position
            fish_position = self.fish.body.position
            vector = prey_position - fish_position  # Taking fish as origin

            # Will generate values between -pi/2 and pi/2 which require adjustment depending on quadrant.
            angle = np.arctan(vector[1] / vector[0])

            if vector[0] < 0 and vector[1] < 0:
                # Generates postiive angle from left x axis clockwise.
                angle += np.pi
            elif vector[1] < 0:
                # Generates negative angle from right x axis anticlockwise.
                angle = angle + (np.pi * 2)
            elif vector[0] < 0:
                # Generates negative angle from left x axis anticlockwise.
                angle = angle + np.pi

            # Angle ends up being between 0 and 2pi as clockwise from right x axis. Same frame as fish angle:
            fish_orientation = (self.fish.body.angle % (2 * np.pi))

            # Normalise so both in same reference frame
            deviation = abs(fish_orientation - angle)
            if deviation > np.pi:
                # Need to account for cases where one angle is very high, while other is very low, as these
                # angles can be close together. Can do this by summing angles and subtracting from 2 pi.
                deviation -= (2 * np.pi)
                deviation = abs(deviation)
            if deviation < self.env_variables["capture_swim_permissive_angle"]:
                valid_capture = True
                self.remove_prey(touched_prey_index)

        if valid_capture:
            self.prey_caught += 1
            self.fish.prey_consumed = True
            self.prey_consumed_this_step = True
            return False
        else:
            self.touched_prey_indices.append(touched_prey_index)
            return True
    
    def prey_touch_body(self, arbiter, space, data):
        touched_prey_index = None
        for i, shp in enumerate(self.prey_shapes):
            if shp == arbiter.shapes[0]:
                touched_prey_index = i
                break
        if touched_prey_index is None: # already removed (prey touched mouth first)
            return True
        self.touched_prey_indices.append(touched_prey_index)
        return True
    
    def remove_prey(self, prey_index):
        self.space.remove(self.prey_shapes[prey_index], self.prey_shapes[prey_index].body)
        del self.prey_shapes[prey_index]
        del self.prey_bodies[prey_index]
        if self.env_variables["prey_reproduction_mode"]:
            del self.prey_ages[prey_index]
        del self.paramecia_gaits[prey_index]
        del self.prey_identifiers[prey_index]
        while True:
            if prey_index in self.touched_prey_indices:
                self.touched_prey_indices.remove(prey_index)
            else:
                break

    def move_predator(self):
        if self.check_predator_at_target() or self.check_predator_outside_walls():
            self.remove_predator()
        else:
            self.predator_body.angle = np.pi / 2 - np.arctan2(
                self.predator_target[0] - self.predator_body.position[0],
                self.predator_target[1] - self.predator_body.position[1])
            self.predator_body.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def touch_predator(self, arbiter, space, data):
        if self.env_variables["test_sensory_system"]:
            self.remove_predator()
        if self.num_steps > self.env_variables['predator_immunity_steps']:
            self.fish.touched_predator = True
            return False
        else:
            return True
        
    def get_predator_angles_distance(self):
        if self.predator_body is None:
            return None, None, None
        predator_position = self.predator_body.position
        fish_position = self.fish.body.position
        distance = np.sqrt(
            (predator_position[0] - fish_position[0]) ** 2 +
            (predator_position[1] - fish_position[1]) ** 2)
        predator_half_angular_size = np.arctan2(self.env_variables['predator_radius'], distance)
        distance -= self.env_variables['predator_radius']
        predator_vector = predator_position - fish_position  # Taking fish as origin
        # Will generate values between -pi/2 and pi/2 which require adjustment depending on quadrant.
        angle = np.arctan2(predator_vector[1], predator_vector[0])
        left_edge = angle + predator_half_angular_size
        right_edge = angle - predator_half_angular_size
        left_edge = np.arctan2(np.sin(left_edge), np.cos(left_edge))  # Normalise to -pi to pi
        right_edge = np.arctan2(np.sin(right_edge), np.cos(right_edge))  # Normalise to -pi to pi

        return left_edge, right_edge, distance        

    def check_fish_proximity_to_walls(self):
        fish_position = self.fish.body.position

        # Check proximity to left wall
        if 0 < fish_position[0] < self.env_variables["predator_distance_from_fish"]:
            left = True
        else:
            left = False

        # Check proximity to right wall
        if self.env_variables["arena_width"] - self.env_variables["predator_distance_from_fish"] < fish_position[0] < \
                self.env_variables["arena_width"]:
            right = True
        else:
            right = False

        # Check proximity to bottom wall
        if self.env_variables["arena_height"] - self.env_variables["predator_distance_from_fish"] < fish_position[1] < \
                self.env_variables["arena_height"]:
            bottom = True
        else:
            bottom = False

        # Check proximity to top wall
        if 0 < fish_position[0] < self.env_variables["predator_distance_from_fish"]:
            top = True
        else:
            top = False

        return left, bottom, right, top

    def select_predator_angle_of_attack(self):
        left, bottom, right, top = self.check_fish_proximity_to_walls()
        if left and top:
            angle_from_fish = self.rng.integers(90, 180)
        elif left and bottom:
            angle_from_fish = self.rng.integers(0, 90)
        elif right and top:
            angle_from_fish = self.rng.integers(180, 270)
        elif right and bottom:
            angle_from_fish = self.rng.integers(270, 360)
        elif left:
            angle_from_fish = self.rng.integers(0, 180)
        elif top:
            angle_from_fish = self.rng.integers(90, 270)
        elif bottom:
            angles = [self.rng.integers(270, 360), self.rng.integers(0, 90)]
            angle_from_fish = self.rng.choice(angles)
        elif right:
            angle_from_fish = self.rng.integers(180, 360)
        else:
            angle_from_fish = self.rng.integers(0, 360)

        angle_from_fish = np.radians(angle_from_fish)
        return angle_from_fish

    def check_fish_not_near_wall(self):
        buffer_region = self.env_variables["predator_radius"] * 1.5
        x_position, y_position = self.fish.body.position[0], self.fish.body.position[1]

        if x_position < buffer_region:
            return True
        elif x_position > self.env_variables["arena_width"] - buffer_region:
            return True
        if y_position < buffer_region:
            return True
        elif y_position > self.env_variables["arena_width"] - buffer_region:
            return True

    def create_predator(self):
        self.predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])
        self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_radius'])
        self.predator_shape.elasticity = 1.0

        fish_position = self.fish.body.position

        if self.env_variables["test_sensory_system"]:
            
            angle_from_fish = np.radians(300)
        else:
            angle_from_fish = self.select_predator_angle_of_attack()
        dy = self.env_variables["predator_distance_from_fish"] * np.cos(angle_from_fish)
        dx = self.env_variables["predator_distance_from_fish"] * np.sin(angle_from_fish)

        x_position = fish_position[0] + dx
        y_position = fish_position[1] + dy

        self.predator_body.position = (x_position, y_position)
        self.predator_target = fish_position

        self.predator_location = (x_position, y_position)

        self.predator_shape.collision_type = 5
        self.predator_shape.filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 2)  # Category 2 objects cant collide with predator

        self.space.add(self.predator_body, self.predator_shape)

    def check_predator_outside_walls(self):
        x_position, y_position = self.predator_body.position[0], self.predator_body.position[1]
        if x_position < 0:
            return True
        elif x_position > self.env_variables["arena_width"]:
            return True
        if y_position < 0:
            return True
        elif y_position > self.env_variables["arena_height"]:
            return True

    def check_predator_at_target(self):
        if (round(self.predator_body.position[0]), round(self.predator_body.position[1])) == (
                round(self.predator_target[0]), round(self.predator_target[1])):
            self.predator_attacks_avoided += 1
            return True
        else:
            return False

    def remove_predator(self, arbiter=None, space=None, data=None):
        if self.predator_body is not None:
            self.space.remove(self.predator_shape, self.predator_shape.body)
            self.predator_shape = None
            self.predator_body = None
            self.predator_location = None
            self.predator_target = None
            if not self.fish.touched_predator:
                self.survived_attack = True
        return False


    def bring_fish_in_bounds(self):
        # Resolve if fish falls out of bounds.
        if self.fish.body.position[0] < 4 or self.fish.body.position[1] < 4 or \
                self.fish.body.position[0] > self.env_variables["arena_width"] - 4 or \
                self.fish.body.position[1] > self.env_variables["arena_height"] - 4:
            new_position = pymunk.Vec2d(np.clip(self.fish.body.position[0], 6, self.env_variables["arena_width"] - 30),
                                        np.clip(self.fish.body.position[1], 6, self.env_variables["arena_height"] - 30))
            self.fish.body.position = new_position

    def get_info(self):
        info_dict = {
            'fish_x': [self.fish.body.position[0]],
            'fish_y': [self.fish.body.position[1]],
            'fish_angle': [self.fish.body.angle],
            'prey_x': [[pr.position[0] for pr in self.prey_bodies]],
            'prey_y': [[pr.position[1] for pr in self.prey_bodies]],
            'predator_x': [self.predator_body.position[0]] if self.predator_body else [0],
            'predator_y': [self.predator_body.position[1]] if self.predator_body else [0],
            'event_consumed_prey': [self.prey_consumed_this_step],
            'event_survived_predator': [self.event_survived_predator],
            'event_captured_by_predator': [self.event_captured_by_predator],
        }
        return info_dict
    
    def step(self, action: int) -> dm_env.TimeStep:
        """Performs a step in the simulation with the given action."""
        self.event_survived_predator = False
        self.event_captured_by_predator = False
        
        self.action_used[action] += 1
        if self._reset_next_step:
            return self.reset()
            
        self.fish.making_capture = False
        self.prey_consumed_this_step = False
        self.last_action = action

        reward = 0
        self.fish.take_action(action)

        done = False

        self.init_predator()
        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.touched_prey_indices = []

    
            if self.fish.making_capture and self.capture_start <= micro_step <= self.capture_end:
                self.fish.capture_possible = True
            else:
                self.fish.capture_possible = False


            self.space.step(self.env_variables['phys_dt'])
            self.move_prey(micro_step)
            if self.predator_body is not None:
                self.move_predator()

            if self.fish.prey_consumed:
                if len(self.prey_shapes) == 0:
                    done = True
                    self.recent_cause_of_death = "Prey-All-Eaten"

                self.fish.prey_consumed = False
            if self.fish.touched_edge:
                self.fish.touched_edge = False

        if self.fish.touched_predator:
            self.event_captured_by_predator = True
            reward += self.env_variables['reward_predator_caught']
            self.survived_attack = False
            self.predator_associated_reward += self.env_variables["reward_predator_caught"]
            self.total_attacks_captured += 1
            self.remove_predator()
            self.fish.touched_predator = False

        if (self.predator_body is None) and self.survived_attack:
            self.event_survived_predator = True
            reward += self.env_variables["reward_predator_avoidance"]
            self.predator_associated_reward += self.env_variables["reward_predator_avoidance"]
            self.total_attacks_avoided += 1
            self.survived_attack = False

        self.bring_fish_in_bounds()

        # Energy level
        energy_reward = self.fish.update_energy_level(self.prey_consumed_this_step)
        reward += energy_reward
        if self.prey_consumed_this_step:
            reward += self.env_variables["reward_consumption"]
            self.consumption_associated_reward += self.env_variables["reward_consumption"]
        self.energy_associated_reward += energy_reward

        self.energy_level_log.append(self.fish.energy_level)
        if self.fish.energy_level < 0:
            print("Fish ran out of energy")
            done = True
            self.recent_cause_of_death = "Starvation"

        # Salt
        if self.env_variables["salt"]:
            self.salt_concentration = self.salt_gradient[int(self.fish.body.position[0]), int(self.fish.body.position[1])]
           
            reward += self.env_variables["reward_salt_factor"] * self.salt_concentration
            self.salt_associated_reward += self.env_variables["reward_salt_factor"] * self.salt_concentration
        else:
            self.salt_concentration = 0

        if self.fish.touched_edge_this_step:
            reward += self.env_variables["reward_wall_touch"]
            self.wall_associated_reward += self.env_variables["reward_wall_touch"]

            self.fish.touched_edge_this_step = False

        if self.env_variables["prey_reproduction_mode"] and self.env_variables["prey_cloud_num"] > 0 and not self.env_variables["test_sensory_system"]:
            self.reproduce_prey()
            self.prey_ages = [age + 1 for age in self.prey_ages]
            for i, age in enumerate(self.prey_ages):
                if age > self.env_variables["prey_safe_duration"] and\
                        self.rng.random(1) < self.env_variables["prey_p_death"]:
                    if not self.check_proximity(self.prey_bodies[i].position, 200):
                        self.remove_prey(i)
                        self.available_prey -= 1

        self.num_steps += 1
        if self.num_steps >= self.env_variables["max_episode_length"]:
            print("Fish ran out of time")
            done = True
            self.recent_cause_of_death = "Time"

        observation = self.get_observation(action, reward)
        if done:
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=observation)
        else:
            return dm_env.transition(reward=reward, observation=observation)

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        len_internal_state = 3
        vis_shape = (len(self.fish.left_eye.interpolated_observation_angles), 3, 2)
        obs_spec = [specs.Array(shape=vis_shape, dtype='float32', name="visual_input"),
                    specs.Array(shape=(len_internal_state,), dtype='float32', name="internal_state")]
        return OAR(observation=obs_spec,
            action=specs.Array(shape=(), dtype=int),
            reward=specs.Array(shape=(), dtype=np.float64),
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int, num_values=self.num_actions, name="action")

    def get_observation(self, action, reward):

        self.arena.red_FOV.update_field_of_view(self.fish.body.position)
        self.arena.uv_FOV.update_field_of_view(self.fish.body.position)
        visual_input = self.resolve_visual_input()
        # print minimal and maximal values of visual input:
        visual_input = visual_input.astype(np.float32)
        # Calculate internal state
        is_in_light = self.fish.body.position[1] > self.dark_row
        internal_state = np.array([is_in_light, self.fish.energy_level, self.salt_concentration], dtype=np.float32)
        return OAR(observation=[visual_input, internal_state],action=action, reward=reward)
    
    def init_predator(self):
        if self.env_variables["test_sensory_system"]:
            if self.num_steps > 10 and not self.tested_predator:
                self.create_predator()
                self.tested_predator = True
        else:

            if self.predator_location is None and \
                    self.rng.random() < self.predator_prob[self.num_steps] and \
                    not self.check_fish_not_near_wall():

                self.create_predator()

    def get_FOV(self, fish_position, max_range, env_width, env_height):
        round_max_range = int(np.round(max_range))
        fish_position = np.round(fish_position).astype(int)

        full_fov_top = fish_position[1] - round_max_range
        full_fov_bottom = fish_position[1] + round_max_range + 1
        full_fov_left = fish_position[0] - round_max_range
        full_fov_right = fish_position[0] + round_max_range + 1

        dim = round_max_range * 2 + 1
        local_fov_top = 0
        local_fov_bottom = dim
        local_fov_left = 0
        local_fov_right = dim

        enclosed_fov_top = full_fov_top
        enclosed_fov_bottom = full_fov_bottom
        enclosed_fov_left = full_fov_left
        enclosed_fov_right = full_fov_right

        if full_fov_top < 0:
            enclosed_fov_top = 0
            local_fov_top = -full_fov_top

        if full_fov_bottom > env_width:
            enclosed_fov_bottom = env_width
            local_fov_bottom = dim - (full_fov_bottom - env_width)

        if full_fov_left < 0:
            enclosed_fov_left = 0
            local_fov_left = -full_fov_left

        if full_fov_right > env_height:
            enclosed_fov_right = env_height
            local_fov_right = dim - (full_fov_right - env_height)

        enclosed_FOV = [enclosed_fov_top, enclosed_fov_bottom, enclosed_fov_left, enclosed_fov_right]
        local_FOV = [local_fov_top, local_fov_bottom, local_fov_left, local_fov_right]

        return enclosed_FOV, local_FOV

    def resolve_visual_input(self):
        # Relative eye positions to center of FOV
        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'],
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'])
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'],
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'])

        if self.predator_body is not None:
            predator_bodies = np.array([self.predator_body.position])
        else:
            predator_bodies = np.array([])

        prey_locations = [i.position for i in self.prey_bodies]
        masked_sediment = self.arena.get_masked_sediment()
        uv_luminance_mask = self.arena.get_uv_luminance_mask()

        if len(prey_locations) > 0:
            prey_locations_array = np.array(prey_locations) - np.array(self.fish.body.position) + self.max_uv_range
        else:
            prey_locations_array = np.array([])

        # check if preditor exists
        if predator_bodies.size > 0:
            predator_left, predator_right, predator_distance = self.get_predator_angles_distance()
        else:
            predator_left, predator_right, predator_distance = np.nan, np.nan, np.nan

        left_photons = self.fish.left_eye.read(masked_sediment, left_eye_pos[0], left_eye_pos[1], self.fish.body.angle, uv_luminance_mask,
                                              prey_locations_array, predator_left, predator_right, predator_distance)
        right_photons = self.fish.right_eye.read(masked_sediment, right_eye_pos[0], right_eye_pos[1], self.fish.body.angle, uv_luminance_mask,
                                              prey_locations_array, predator_left, predator_right, predator_distance)

        observation = np.dstack((left_photons, right_photons))

        return observation
