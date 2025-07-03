import copy
import random
import numpy as np
import pymunk
import dm_env
from dm_env import specs
import time

from Environment.Board.drawing_board import DrawingBoard
from Environment.Fish.fish import Fish
from acme.wrappers import observation_action_reward

OAR = observation_action_reward.OAR


class BaseEnvironment(dm_env.Environment):
    """A base class to represent environments, for extension to ProjectionEnvironment, VVR and Naturalistic
    environment classes."""

    def __init__(self, env_variables):

        super().__init__()

        self.env_variables = env_variables
        self.num_actions = self.env_variables['num_actions']

        
        print("defining drawing board")
        self.max_uv_range = np.absolute(np.log(0.001) / self.env_variables["light_decay_rate"])

        # self.board = DrawingBoard(arena_width=self.env_variables['arena_width'],
        #                           arena_height=self.env_variables['arena_height'],
        #                           uv_light_decay_rate=self.env_variables['light_decay_rate'],
        #                           red_light_decay_rate=self.env_variables['light_decay_rate'],
        #                           photoreceptor_rf_size=max_photoreceptor_rf_size,
        #                           prey_radius=self.env_variables['prey_radius'],
        #                           predator_radius=self.env_variables['predator_radius'],
        #                           visible_scatter=self.env_variables['background_brightness'],
        #                           dark_light_ratio=self.env_variables['dark_light_ratio'],
        #                           dark_gain=self.env_variables['dark_gain'],
        #                           light_gain=self.env_variables['light_gain'],
        #                           light_gradient=light_gradient,
        #                           max_red_range=max_red_range,
        #                           max_uv_range=self.max_uv_range,
        #                           red_object_intensity=self.env_variables["background_point_intensity"],
        #                           sediment_sigma=self.env_variables["sediment_sigma"],
        #                           #red2_object_intensity=self.env_variables["background_point_intensity"],
        #                           )

        self.board = DrawingBoard(self.env_variables)
        print('defining physics')
        self.dark_col = int(self.env_variables['arena_width'] * self.env_variables['dark_light_ratio'])
        if self.dark_col == 0:  # Fixes bug with left wall always being invisible.
            self.dark_col = -1

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = self.env_variables['drag']

        self.fish = Fish(board=self.board,
                         env_variables=env_variables,
                         max_uv_range=self.max_uv_range,
                         )        



        self.stimuli_information = {}
 
        if self.env_variables["salt"]:
            self.salt_gradient = None
            self.xp, self.yp = np.arange(self.env_variables['arena_width']), np.arange(
                self.env_variables['arena_height'])
        self.salt_location = None


        # For currents (new simulation):
        self.impulse_vector_field = None
        self.coordinates_in_current = None  # May be used to provide efficient checking. Although vector comp probably faster.
        #self.create_current()
        self.capture_fraction = int(
            self.env_variables["phys_steps_per_sim_step"] * self.env_variables['fraction_capture_permitted'])
        self.capture_start = 1  # int((self.env_variables['phys_steps_per_sim_step'] - self.capture_fraction) / 2)
        self.capture_end = self.capture_start + self.capture_fraction

        #self.paramecia_distances = []

        self.space.add(self.fish.body, self.fish.mouth, self.fish.head, self.fish.tail)
        self.prey_shapes = []
        self.sand_grain_shapes = []
        self.prey_cloud_wall_shapes = []
        self.predator_shape = None
        self.energy_associated_reward = 0
        self.action_associated_reward = 0
        self.salt_associated_reward = 0
        self.predator_associated_reward = 0
        self.wall_associated_reward = 0
        self.sand_grain_associated_reward = 0


        self.create_walls()
        #self.reset()

        self.set_collisions()

        self.continuous_actions = False
        self._reset_next_step = True
        self.action_used = np.zeros(12)



    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.tested_predator = False
        self.num_steps = 0
        self.fish.stress = 1
        self.fish.touched_edge_this_step = False
        self.prey_caught = 0
        self.predator_attacks_avoided = 0
        self.sand_grains_bumped = 0
        self.energy_level_log = []
        self.salt_concentration = 0
        self.switch_step = None
        # New energy system:
        self.fish.energy_level = 1

        # Reset salt gradient
        if self.env_variables["salt"]:
            self.reset_salt_gradient()
        else:
            self.salt_location = [np.nan, np.nan]

        self.clear_environmental_features()
        self.board.reset()

        self.mask_buffer = []
        self.action_buffer = []
        self.position_buffer = []
        self.fish_angle_buffer = []

        self.failed_capture_attempts = 0
        self.in_light_history = []
        if self.env_variables['test_sensory_system']:
            self.fish.body.position = (self.env_variables['arena_width'] / 2, self.env_variables['arena_height'] / 2)
            self.fish.body.angle = 0
        else:
            self.fish.body.position = (np.random.randint(self.env_variables['fish_mouth_radius'] + 40,
                                                        self.env_variables['arena_width'] - (self.env_variables[
                                                                                                'fish_mouth_radius'] + 40)),
                                    np.random.randint(self.env_variables['fish_mouth_radius'] + 40,
                                                        self.env_variables['arena_height'] - (self.env_variables[
                                                                                                'fish_mouth_radius'] + 40)))
            self.fish.body.angle = np.random.random() * 2 * np.pi

        self.fish.body.velocity = (0, 0)
        if self.env_variables["current_setting"]:
            self.impulse_vector_field *= np.random.choice([-1, 1], size=1, p=[0.5, 0.5]).astype(float)
        self.fish.capture_possible = False

        if self.env_variables["differential_prey"]:
            self.prey_cloud_locations = [
                [np.random.randint(
                    low=120 + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                    high=self.env_variables['arena_width'] - (
                            self.env_variables['prey_radius'] + self.env_variables[
                        'fish_mouth_radius']) - 120),
                 np.random.randint(
                     low=120 + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                     high=self.env_variables['arena_height'] - (
                             self.env_variables['prey_radius'] + self.env_variables[
                         'fish_mouth_radius']) - 120)]
                for cloud in range(int(self.env_variables["prey_cloud_num"]))]

            self.sand_grain_cloud_locations = [
                [np.random.randint(
                    low=120 + self.env_variables['sand_grain_radius'] + self.env_variables['fish_mouth_radius'],
                    high=self.env_variables['arena_width'] - (
                            self.env_variables['sand_grain_radius'] + self.env_variables[
                        'fish_mouth_radius']) - 120),
                    np.random.randint(
                        low=120 + self.env_variables['sand_grain_radius'] + self.env_variables['fish_mouth_radius'],
                        high=self.env_variables['arena_height'] - (
                                self.env_variables['sand_grain_radius'] + self.env_variables[
                            'fish_mouth_radius']) - 120)]
                for cloud in range(int(self.env_variables["sand_grain_num"]))]

            if "fixed_prey_distribution" in self.env_variables:
                if self.env_variables["fixed_prey_distribution"]:
                    x_locations = np.linspace(
                        120 + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                        self.env_variables['arena_width'] - (
                                self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius']) - 120,
                        np.ceil(self.env_variables["prey_cloud_num"] ** 0.5))
                    y_locations = np.linspace(
                        120 + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                        self.env_variables['arena_width'] - (
                                self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius']) - 120,
                        np.ceil(self.env_variables["prey_cloud_num"] ** 0.5))

                    self.prey_cloud_locations = np.concatenate((np.expand_dims(x_locations, 1),
                                                                np.expand_dims(y_locations, 1)), axis=1)
                    self.prey_cloud_locations = self.prey_cloud_locations[:self.env_variables["prey_cloud_num"]]

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

        for i in range(self.env_variables['sand_grain_num']):
            self.create_sand_grain()

        self.impulse_against_fish_previous_step = None
        self.recent_cause_of_death = None
        self.available_prey = self.env_variables["prey_num"]
        self.vector_agreement = []

        self.total_predators = 0
        self.total_predators_survived = 0

        self.predator_body = None
        self.predator_shape = None
        self.predator_target = None

        self.last_action = None
        self.prey_consumed_this_step = False

        self.touched_sand_grain = False
        self.survived_attack = False

        # For Reward tracking (debugging)
        print(f"""REWARD CONTRIBUTIONS:        
              Energy: {self.energy_associated_reward}
              Action: {self.action_associated_reward}
              Salt: {self.salt_associated_reward}
              Predator: {self.predator_associated_reward}
              Wall: {self.wall_associated_reward}
              Sand grain: {self.sand_grain_associated_reward}
              """)
        print(f"actions used: {self.action_used / np.sum(self.action_used)}")
        self.energy_associated_reward = 0
        self.action_associated_reward = 0
        self.salt_associated_reward = 0
        self.predator_associated_reward = 0
        self.wall_associated_reward = 0
        self.sand_grain_associated_reward = 0
        self.action_used = np.zeros(12)

        self.num_steps_prey_available = 0
        return dm_env.restart(self.get_observation(action=0, reward=0.))

    def set_collisions(self):
        """Specifies the collisions that occur in the Pymunk simulation."""

        # Collision Types:
        # 1: Edge
        # 2: Prey
        # 3: Fish mouth
        # 4: Sand grains
        # 5: Predator
        # 6: Fish body
        # 7: Prey cloud wall

        self.col = self.space.add_collision_handler(2, 3)
        self.col.begin = self.touch_prey

        self.pred_col = self.space.add_collision_handler(5, 3)
        self.pred_col.begin = self.touch_predator
        self.pred_col2 = self.space.add_collision_handler(5, 6)
        self.pred_col2.begin = self.touch_predator

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_wall

        self.edge_pred_col = self.space.add_collision_handler(1, 5)
        self.edge_pred_col.begin = self.remove_predator

        self.grain_fish_col = self.space.add_collision_handler(3, 4)
        self.grain_fish_col.begin = self.touch_grain

        # to prevent predators from knocking out prey  or static grains
        self.grain_pred_col = self.space.add_collision_handler(4, 5)
        self.grain_pred_col.begin = self.no_collision
        self.prey_pred_col = self.space.add_collision_handler(2, 5)
        self.prey_pred_col.begin = self.no_collision

        # To prevent the differential wall being hit by fish
        self.fish_prey_wall = self.space.add_collision_handler(3, 7)
        self.fish_prey_wall.begin = self.no_collision
        self.fish_prey_wall2 = self.space.add_collision_handler(6, 7)
        self.fish_prey_wall2.begin = self.no_collision
        self.pred_prey_wall2 = self.space.add_collision_handler(5, 7)
        self.pred_prey_wall2.begin = self.no_collision

    def draw_walls_and_sediment(self):
        """Draws the walls and background sediment on the drawing board, which is used for computing visual inputs."""
        self.board.erase()
        #self.board.draw_walls()
        self.board.draw_sediment()

    def clear_environmental_features(self):
        """Removes all prey, predators, and sand grains from simulation"""
        for i, shp in enumerate(self.prey_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.sand_grain_shapes):
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
        self.prey_ages = []
        self.total_prey_created = 0

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []


    def get_prey_within_visual_field(self, max_angular_deviation, max_distance):
        prey_near = self.check_proximity_all_prey(sensing_distance=max_distance)
        fish_prey_incidence = self.get_fish_prey_incidence()
        within_visual_field = np.absolute(fish_prey_incidence) < max_angular_deviation

        prey_in_visual_field = prey_near * within_visual_field

        return prey_in_visual_field

    def reproduce_prey(self):
        num_prey = len(self.prey_bodies)
        # p_prey_birth = self.env_variables["birth_rate"] / (
        #         num_prey * self.env_variables["birth_rate_current_pop_scaling"])
        p_prey_birth = self.env_variables["birth_rate"] * (self.env_variables["prey_num"] - num_prey)
        for cloud in self.prey_cloud_locations:
            if np.random.rand(1) < p_prey_birth:
                if not self.check_proximity(cloud, self.env_variables["prey_cloud_region_size"]):
                    new_location = (
                        np.random.randint(low=cloud[0] - (self.env_variables["prey_cloud_region_size"] / 2),
                                          high=cloud[0] + (self.env_variables["prey_cloud_region_size"] / 2)),
                        np.random.randint(low=cloud[1] - (self.env_variables["prey_cloud_region_size"] / 2),
                                          high=cloud[1] + (self.env_variables["prey_cloud_region_size"] / 2))
                    )
                    self.create_prey(new_location)
                    self.available_prey += 1

    def reset_salt_gradient(self, salt_source=None):
        if salt_source is None:
            salt_source_x = np.random.randint(0, self.env_variables['arena_width'] - 1)
            salt_source_y = np.random.randint(0, self.env_variables['arena_height'] - 1)
        else:
            salt_source_x = salt_source[0]
            salt_source_y = salt_source[1]

        self.salt_location = [salt_source_x, salt_source_y]
        salt_distance = (((salt_source_x - self.xp[:, None]) ** 2 + (
                salt_source_y - self.yp[None, :]) ** 2) ** 0.5)  # Measure of distance from source at every point.
        self.salt_gradient = np.exp(-self.env_variables["salt_concentration_decay"] * salt_distance) 

    def build_prey_cloud_walls(self):
        for i in self.prey_cloud_locations:
            wall_edges = [
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] - 150), (i[0] - 150, i[1] + 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] + 150), (i[0] + 150, i[1] + 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] + 150, i[1] + 150), (i[0] + 150, i[1] - 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] - 150), (i[0] + 150, i[1] - 150), 1)
            ]
            for s in wall_edges:
                s.friction = 1.
                s.group = 1
                s.collision_type = 7
                s.color = (0, 0, 0)
                self.space.add(s)
                self.prey_cloud_wall_shapes.append(s)

    def create_walls(self):
        # wall_width = 1
        wall_width = 5  # self.env_variables['eyes_biasx']
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
            s.color = (1, 0, 0)
            self.space.add(s)

    @staticmethod
    def no_collision(arbiter, space, data):
        return False

    def touch_wall(self, arbiter, space, data):
        if not self.env_variables["wall_reflection"]:
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
        self.prey_bodies[-1].angle = np.random.uniform(0, np.pi * 2)
        if prey_position is None:
            if not self.env_variables["differential_prey"]:
                self.prey_bodies[-1].position = (
                    np.random.randint(
                        self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'] + 40,
                        self.env_variables['arena_width'] - (
                                self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'] +
                                40)),
                    np.random.randint(
                        self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'] + 40,
                        self.env_variables['arena_height'] - (
                                self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'] +
                                40)))
            else:
                cloud = random.choice(self.prey_cloud_locations)
                self.prey_bodies[-1].position = (
                    np.random.randint(low=cloud[0] - (self.env_variables["prey_cloud_region_size"] / 2),
                                      high=cloud[0] + (self.env_variables["prey_cloud_region_size"] / 2)),
                    np.random.randint(low=cloud[1] - (self.env_variables["prey_cloud_region_size"] / 2),
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
                np.random.choice([0, 1, 2], 1, p=[1 - (self.env_variables["p_fast"] + self.env_variables["p_slow"]),
                                                  self.env_variables["p_slow"],
                                                  self.env_variables["p_fast"]])[0])
            if self.env_variables["prey_reproduction_mode"]:
                self.prey_ages.append(0)
        else:

            self.paramecia_gaits.append(int(prey_gait))
            if self.env_variables["prey_reproduction_mode"]:
                self.prey_ages.append(int(prey_age))

        self.prey_shapes[-1].color = (0, 0, 1)
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

        fish_prey_distances = ((fish_prey_vectors[:, 0] ** 2) + (fish_prey_vectors[:, 1] ** 2) ** 0.5)
        within_range = fish_prey_distances < sensing_distance
        return within_range

    def get_fish_prey_incidence(self):
        fish_orientation = self.fish.body.angle
        fish_position = np.expand_dims(np.array(self.fish.body.position), axis=0)
        paramecium_positions = np.array([pr.position for pr in self.prey_bodies])
        fish_orientation = np.array([fish_orientation])

        fish_orientation_sign = ((fish_orientation >= 0) * 1) + ((fish_orientation < 0) * -1)

        # Remove full orientations (so is between -2pi and 2pi
        fish_orientation %= 2 * np.pi * fish_orientation_sign

        # Convert to positive scale between 0 and 2pi
        fish_orientation[fish_orientation < 0] += 2 * np.pi

        fish_prey_vectors = paramecium_positions - fish_position

        # Adjust according to quadrents.
        fish_prey_angles = np.arctan(fish_prey_vectors[:, 1] / fish_prey_vectors[:, 0])

        #   Generates positive angle from left x axis clockwise.
        # UL quadrent
        in_ul_quadrent = (fish_prey_vectors[:, 0] < 0) * (fish_prey_vectors[:, 1] > 0)
        fish_prey_angles[in_ul_quadrent] += np.pi
        # BR quadrent
        in_br_quadrent = (fish_prey_vectors[:, 0] > 0) * (fish_prey_vectors[:, 1] < 0)
        fish_prey_angles[in_br_quadrent] += (np.pi * 2)
        # BL quadrent
        in_bl_quadrent = (fish_prey_vectors[:, 0] < 0) * (fish_prey_vectors[:, 1] < 0)
        fish_prey_angles[in_bl_quadrent] += np.pi

        # Angle ends up being between 0 and 2pi as clockwise from right x-axis. Same frame as fish angle:
        fish_prey_incidence = np.expand_dims(np.array([fish_orientation]), 1) - fish_prey_angles

        fish_prey_incidence[fish_prey_incidence > np.pi] %= np.pi
        fish_prey_incidence[fish_prey_incidence < -np.pi] %= -np.pi

        return fish_prey_incidence

    def move_prey(self, micro_step):
        if len(self.prey_bodies) == 0:
            return

        # Generate impulses
        impulse_types = [0, self.env_variables["slow_impulse_paramecia"], self.env_variables["fast_impulse_paramecia"]]
        impulses = [impulse_types[gait] for gait in self.paramecia_gaits]

        # Do once per step.
        if micro_step == 0:
            gaits_to_switch = np.random.choice([0, 1], len(self.prey_shapes),
                                               p=[1 - self.env_variables["p_switch"], self.env_variables["p_switch"]])
            switch_to = np.random.choice([0, 1, 2], len(self.prey_shapes),
                                         p=[1 - (self.env_variables["p_slow"] + self.env_variables["p_fast"]),
                                            self.env_variables["p_slow"], self.env_variables["p_fast"]])
            self.paramecia_gaits = [switch_to[i] if gaits_to_switch[i] else old_gait for i, old_gait in
                                    enumerate(self.paramecia_gaits)]

            # Angles of change
            angle_changes = np.random.uniform(-self.env_variables['prey_max_turning_angle'],
                                              self.env_variables['prey_max_turning_angle'],
                                              len(self.prey_shapes))

            # Large angle changes
            large_turns = np.random.uniform(-np.pi, np.pi, len(self.prey_shapes))
            turns_implemented = np.random.choice([0, 1], len(self.prey_shapes), p=[1 - self.env_variables["p_reorient"],
                                                                                   self.env_variables["p_reorient"]])
            angle_changes = angle_changes + (large_turns * turns_implemented)

            self.prey_within_range = self.check_proximity_all_prey(self.env_variables["prey_sensing_distance"])

        for i, prey_body in enumerate(self.prey_bodies):
            if self.prey_within_range[i]:
                # Motion from fluid dynamics
                if self.env_variables["prey_fluid_displacement"]:
                    distance_vector = prey_body.position - self.fish.body.position
                    distance = (distance_vector[0] ** 2 + distance_vector[1] ** 2) ** 0.5
                    distance_scaling = np.exp(-distance)

                    original_angle = copy.copy(prey_body.angle)
                    prey_body.angle = self.fish.body.angle + np.random.uniform(-1, 1)
                    impulse_for_prey = (self.get_last_action_magnitude() / self.env_variables["known_max_fish_i"]) * \
                                       self.env_variables["displacement_scaling_factor"] * distance_scaling

                    prey_body.apply_impulse_at_local_point((impulse_for_prey, 0))
                    prey_body.angle = original_angle

                # Motion from prey escape
                if self.env_variables["prey_jump"] and np.random.choice([0, 1], size=1,
                                                                        p=[1 - self.env_variables["p_escape"] /
                                                                           self.env_variables[
                                                                               "phys_steps_per_sim_step"],
                                                                           self.env_variables["p_escape"] /
                                                                           self.env_variables[
                                                                               "phys_steps_per_sim_step"]])[0] == 1:
                    prey_body.apply_impulse_at_local_point((self.env_variables["jump_impulse_paramecia"], 0))

            else:
                if micro_step == 0:
                    prey_body.angle = prey_body.angle + angle_changes[i]

                prey_body.apply_impulse_at_local_point((impulses[i], 0))

    def touch_prey(self, arbiter, space, data):
        valid_capture = False
        if self.fish.capture_possible:
            for i, shp in enumerate(self.prey_shapes):
                if shp == arbiter.shapes[0]:
                    # Check if angles line up.
                    prey_position = self.prey_bodies[i].position
                    fish_position = self.fish.body.position
                    vector = prey_position - fish_position  # Taking fish as origin

                    # Will generate values between -pi/2 and pi/2 which require adjustment depending on quadrant.
                    angle = np.arctan(vector[1] / vector[0])

                    if vector[0] < 0 and vector[1] < 0:
                        # Generates postiive angle from left x axis clockwise.
                        # print("UL quadrent")
                        angle += np.pi
                    elif vector[1] < 0:
                        # Generates negative angle from right x axis anticlockwise.
                        # print("UR quadrent.")
                        angle = angle + (np.pi * 2)
                    elif vector[0] < 0:
                        # Generates negative angle from left x axis anticlockwise.
                        # print("BL quadrent.")
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
                    if deviation < self.env_variables["capture_angle_deviation_allowance"]:
                        valid_capture = True
                        self.remove_prey(i)
                    else:
                        self.failed_capture_attempts += 1

            if valid_capture:
                self.prey_caught += 1
                self.fish.prey_consumed = True
                self.prey_consumed_this_step = True

            return False
        else:
            self.failed_capture_attempts += 1
            return True

    def remove_prey(self, prey_index):
        self.space.remove(self.prey_shapes[prey_index], self.prey_shapes[prey_index].body)
        del self.prey_shapes[prey_index]
        del self.prey_bodies[prey_index]
        del self.prey_ages[prey_index]
        del self.paramecia_gaits[prey_index]
        del self.prey_identifiers[prey_index]

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
        if self.num_steps > self.env_variables['immunity_steps']:
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
        if 0 < fish_position[0] < self.env_variables["distance_from_fish"]:
            left = True
        else:
            left = False

        # Check proximity to right wall
        if self.env_variables["arena_width"] - self.env_variables["distance_from_fish"] < fish_position[0] < \
                self.env_variables["arena_width"]:
            right = True
        else:
            right = False

        # Check proximity to bottom wall
        if self.env_variables["arena_height"] - self.env_variables["distance_from_fish"] < fish_position[1] < \
                self.env_variables["arena_height"]:
            bottom = True
        else:
            bottom = False

        # Check proximity to top wall
        if 0 < fish_position[0] < self.env_variables["distance_from_fish"]:
            top = True
        else:
            top = False

        return left, bottom, right, top

    def select_predator_angle_of_attack(self):
        left, bottom, right, top = self.check_fish_proximity_to_walls()
        if left and top:
            angle_from_fish = random.randint(90, 180)
        elif left and bottom:
            angle_from_fish = random.randint(0, 90)
        elif right and top:
            angle_from_fish = random.randint(180, 270)
        elif right and bottom:
            angle_from_fish = random.randint(270, 360)
        elif left:
            angle_from_fish = random.randint(0, 180)
        elif top:
            angle_from_fish = random.randint(90, 270)
        elif bottom:
            angles = [random.randint(270, 360), random.randint(0, 90)]
            angle_from_fish = random.choice(angles)
        elif right:
            angle_from_fish = random.randint(180, 360)
        else:
            angle_from_fish = random.randint(0, 360)

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

    def load_predator(self, predator_position, predator_orientation, predator_target):

        self.predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])
        self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_radius'])
        self.predator_shape.elasticity = 1.0

        self.predator_body.position = (predator_position[0], predator_position[1])
        self.predator_body.angle = predator_orientation
        self.predator_target = predator_target

        self.predator_shape.color = (0, 1, 0)
        self.predator_location = (predator_position[0], predator_position[1])
        self.predator_shape.collision_type = 5
        self.predator_shape.filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # Category 2 objects cant collide with predator

        self.space.add(self.predator_body, self.predator_shape)

    def create_predator(self):
        self.total_predators += 1
        self.predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])
        self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_radius'])
        self.predator_shape.elasticity = 1.0

        fish_position = self.fish.body.position

        if self.env_variables["test_sensory_system"]:
            # choose from 0, 90, 180, 270 degrees
            angle_from_fish = np.radians(300)#np.radians(np.random.choice([90, 180, 270]))
            print("Predator angle from fish: ", np.degrees(angle_from_fish))
        else:
            angle_from_fish = self.select_predator_angle_of_attack()
        dy = self.env_variables["distance_from_fish"] * np.cos(angle_from_fish)
        dx = self.env_variables["distance_from_fish"] * np.sin(angle_from_fish)

        x_position = fish_position[0] + dx
        y_position = fish_position[1] + dy

        self.predator_body.position = (x_position, y_position)
        self.predator_target = fish_position

        self.predator_shape.color = (0, 1, 0)
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
        else:
            pass

    def create_sand_grain(self):
        self.sand_grain_bodies.append(
            pymunk.Body(self.env_variables['sand_grain_mass'], self.env_variables['sand_grain_inertia']))
        self.sand_grain_shapes.append(
            pymunk.Circle(self.sand_grain_bodies[-1], self.env_variables['sand_grain_radius']))
        self.sand_grain_shapes[-1].elasticity = 1.0

        if not self.env_variables["differential_prey"]:
            self.sand_grain_bodies[-1].position = (
                np.random.randint(self.env_variables['sand_grain_radius'] + self.env_variables['fish_mouth_radius'],
                                  self.env_variables['arena_width'] - (
                                          self.env_variables['sand_grain_radius'] + self.env_variables[
                                      'fish_mouth_radius'])),
                np.random.randint(self.env_variables['sand_grain_radius'] + self.env_variables['fish_mouth_radius'],
                                  self.env_variables['arena_height'] - (
                                          self.env_variables['sand_grain_radius'] + self.env_variables[
                                      'fish_mouth_radius'])))
        else:
            cloud = random.choice(self.sand_grain_cloud_locations)
            self.sand_grain_bodies[-1].position = (
                np.random.randint(low=cloud[0] - (self.env_variables["prey_cloud_region_size"] / 2),
                                  high=cloud[0] + (self.env_variables["prey_cloud_region_size"] / 2)),
                np.random.randint(low=cloud[1] - (self.env_variables["prey_cloud_region_size"] / 2),
                                  high=cloud[1] + (self.env_variables["prey_cloud_region_size"] / 2))
            )

        self.sand_grain_shapes[-1].color = (0, 0, 1)

        self.sand_grain_shapes[-1].collision_type = 4
        self.sand_grain_shapes[-1].filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.sand_grain_bodies[-1], self.sand_grain_shapes[-1])

    def touch_grain(self, arbiter, space, data):
        self.fish.touched_sand_grain = True

        if self.last_action == 3:
            self.sand_grains_bumped += 1

    def get_last_action_magnitude(self):
        return self.fish.prev_action_impulse * self.env_variables['displacement_scaling_factor']
        # Scaled down both for mass effects and to make it possible for the prey to be caught.

    def displace_sand_grains(self):
        for i, body in enumerate(self.sand_grain_bodies):
            if self.check_proximity(self.sand_grain_bodies[i].position,
                                    self.env_variables['sand_grain_displacement_distance']):
                self.sand_grain_bodies[i].angle = self.fish.body.angle + np.random.uniform(-1, 1)
                self.sand_grain_bodies[i].apply_impulse_at_local_point(
                    (self.get_last_action_magnitude(), 0))

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
        }
        return info_dict
    def step(self, action: int) -> dm_env.TimeStep:
        
        self.action_used[action] += 1
        t0 = time.time()
        if self._reset_next_step:
            return self.reset()
            
        self.fish.making_capture = False
        self.prey_consumed_this_step = False
        self.last_action = action
        self.fish.touched_sand_grain = False

        # Visualisation
        self.action_buffer.append(action)
        self.fish_angle_buffer.append(self.fish.body.angle)
        self.position_buffer.append(np.array(self.fish.body.position))

        reward = self.fish.take_action(action)

        # For Reward tracking (debugging)
        self.action_associated_reward += reward

        # For impulse direction logging (current opposition metric)
        self.fish.impulse_vector_x = self.fish.prev_action_impulse * np.sin(self.fish.body.angle)
        self.fish.impulse_vector_y = self.fish.prev_action_impulse * np.cos(self.fish.body.angle)
        done = False

        # Change internal state variables
        self.fish.stress = self.fish.stress * self.env_variables['stress_compound']
        if self.predator_body is not None:
            self.fish.stress += 0.5

        self.init_predator()

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.move_prey(micro_step)
            self.displace_sand_grains()

            #if self.env_variables["current_setting"]:
            #    self.bring_fish_in_bounds()
            #    self.resolve_currents(micro_step)
            if self.fish.making_capture and self.capture_start <= micro_step <= self.capture_end:
                self.fish.capture_possible = True
            else:
                self.fish.capture_possible = False

            if self.predator_body is not None:
                self.move_predator()

            self.space.step(self.env_variables['phys_dt'])

            if self.fish.prey_consumed:
                if len(self.prey_shapes) == 0:
                    done = True
                    self.recent_cause_of_death = "Prey-All-Eaten"

                self.fish.prey_consumed = False
            if self.fish.touched_edge:
                self.fish.touched_edge = False

        if self.fish.touched_predator:
            print("Fish eaten by predator")
            reward -= self.env_variables['predator_cost']
            self.survived_attack = False
            self.predator_associated_reward -= self.env_variables["predator_cost"]
            self.remove_predator()
            self.fish.touched_predator = False

            # self.recent_cause_of_death = "Predator"
            # done = True

        if (self.predator_body is None) and self.survived_attack:
            print("Survived attack...")
            reward += self.env_variables["predator_avoidance_reward"]
            self.predator_associated_reward += self.env_variables["predator_cost"]
            self.survived_attack = False
            self.total_predators_survived += 1

        if self.fish.touched_sand_grain:
            reward -= self.env_variables["sand_grain_touch_penalty"]
            self.sand_grain_associated_reward -= self.env_variables["sand_grain_touch_penalty"]

        # Relocate fish (Assay mode only)
        # if self.relocate_fish is not None:
        #     if self.relocate_fish[self.num_steps]:
        #         self.transport_fish(self.relocate_fish[self.num_steps])

        self.bring_fish_in_bounds()

        # Energy level
        if self.env_variables["energy_state"]:
            old_reward = reward
            reward = self.fish.update_energy_level(reward, self.prey_consumed_this_step)
            self.energy_associated_reward += reward - old_reward

            self.energy_level_log.append(self.fish.energy_level)
            if self.fish.energy_level < 0:
                print("Fish ran out of energy")
                done = True
                self.recent_cause_of_death = "Starvation"

        # Salt health
        if self.env_variables["salt"]:
            self.salt_concentration = self.salt_gradient[int(self.fish.body.position[0]), int(self.fish.body.position[1])]
            # self.fish.salt_health = self.fish.salt_health + self.env_variables["salt_recovery"] - self.salt_damage
            # if self.fish.salt_health > 1.0:
            #     self.fish.salt_health = 1.0
            # if self.fish.salt_health < 0:
            #     pass
                # done = True
                # self.recent_cause_of_death = "Salt"

            
            reward -= self.env_variables["salt_reward_penalty"] * self.salt_concentration
            self.salt_associated_reward -= self.env_variables['salt_reward_penalty'] * self.salt_concentration
        else:
            self.salt_concentration = 0

        if self.fish.touched_edge_this_step:
            reward -= self.env_variables["wall_touch_penalty"]
            self.wall_associated_reward -= self.env_variables["wall_touch_penalty"]

            self.fish.touched_edge_this_step = False

        if self.env_variables["prey_reproduction_mode"] and self.env_variables["differential_prey"] and not self.env_variables["test_sensory_system"]:
            self.reproduce_prey()
            self.prey_ages = [age + 1 for age in self.prey_ages]
            for i, age in enumerate(self.prey_ages):
                if age > self.env_variables["prey_safe_duration"] and\
                        np.random.rand(1) < self.env_variables["p_prey_death"]:
                    if not self.check_proximity(self.prey_bodies[i].position, 200):
                        self.remove_prey(i)
                        self.available_prey -= 1

        # Log whether fish in light
        self.in_light_history.append(self.fish.body.position[0] > self.dark_col)

        # Log steps where prey in visual field.
        if len(self.prey_bodies) > 0:
            prey_in_visual_field = self.get_prey_within_visual_field(max_angular_deviation=2.2, max_distance=100)
            num_prey_close = np.sum(prey_in_visual_field * 1)
            if num_prey_close > 0:
                self.num_steps_prey_available += 1

        self.num_steps += 1
        if self.num_steps >= self.env_variables["max_epLength"]:
            print("Fish ran out of time")
            done = True
            self.recent_cause_of_death = "Time"

        # Drawing the features visible at this step:
        self.draw_walls_and_sediment()


        # if self.assay_run_version == "Original" and self.num_steps > 2:  # Temporal conditional stops assay buffer size errors.
        #     if self.check_condition_met():
        #         print(f"Split condition met at step: {self.num_steps}")
        #         done = True
        #         self.switch_step = self.num_steps
        observation = self.get_observation(action, reward)
        t_step = time.time() - t0
        #print(f"Step time: {t_step}")
        if done:
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=observation)
        else:
            return dm_env.transition(reward=reward, observation=observation)

#        observation, full_masked_image = self.resolve_visual_input()
#
#        return observation, reward, internal_state, done, full_masked_image


    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        len_internal_state = self.env_variables['in_light'] + self.env_variables['stress'] + \
                                self.env_variables['energy_state'] + self.env_variables['salt']
        if len_internal_state == 0:
            len_internal_state = 1
        vis_shape = (len(self.fish.left_eye.interpolated_observation), 3, 2)
        #obs_spec = specs.Array(shape=vis_shape, dtype=np.float32, name="visual_input")
        obs_spec = [specs.Array(shape=vis_shape, dtype='float32', name="visual_input"),
                    specs.Array(shape=(len_internal_state,), dtype='float32', name="internal_state")]
        return OAR(observation=obs_spec,
            action=specs.Array(shape=(), dtype=int),
            reward=specs.Array(shape=(), dtype=np.float64),
        )

#        return specs.Array(shape=vis_shape, dtype='float32', name="visual_input")
#        return [specs.Array(shape=vis_shape, dtype='float32', name="visual_input"),
#                specs.Array(shape=(1, len_internal_state), dtype=np.float32, name="internal_state")]
    

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int, num_values=self.num_actions, name="action")

    def get_observation(self, action, reward):

        self.board.FOV.update_field_of_view(self.fish.body.position)
        visual_input = self.resolve_visual_input()
        # print minimal and maximal values of visual input:
        visual_input = visual_input.astype(np.float32)
        # Calculate internal state
        internal_state = []
        internal_state_order = []
        if self.env_variables['in_light']:
            internal_state.append(self.fish.body.position[0] > self.dark_col)
            internal_state_order.append("in_light")
        if self.env_variables['stress']:
            internal_state.append(self.fish.stress)
            internal_state_order.append("stress")
        if self.env_variables['energy_state']:
            internal_state.append(self.fish.energy_level)
            # print(self.fish.energy_level)
            internal_state_order.append("energy_state")
        if self.env_variables['salt']:
            
            internal_state.append(self.salt_concentration)

            internal_state_order.append("salt")
        if len(internal_state) == 0:
            internal_state.append(0)
        internal_state = np.array(internal_state, dtype=np.float32)
        return OAR(observation=[visual_input, internal_state],action=action, reward=reward)
    
    def init_predator(self):
        if self.env_variables["test_sensory_system"]:
            if self.num_steps > 10 and not self.tested_predator:
                self.create_predator()
                self.tested_predator = True
        else:

            if self.predator_location is None and \
                    np.random.rand(1) < self.env_variables["probability_of_predator"] and \
                    self.num_steps > self.env_variables['immunity_steps'] and \
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
                'eyes_biasx'],# + self.board.max_red_range,
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'])# + self.board.max_red_range)
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'],# + self.board.max_red_range,
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'])# + self.board.max_red_range)

        if self.predator_body is not None:
            predator_bodies = np.array([self.predator_body.position])
        else:
            predator_bodies = np.array([])

        prey_locations = [i.position for i in self.prey_bodies]
        sand_grain_locations = [i.position for i in self.sand_grain_bodies]
        # full_masked_image, lum_mask = self.board.get_masked_pixels(np.array(self.fish.body.position),
        #                                                            np.array(prey_locations + sand_grain_locations),
        #                                                            predator_bodies)
        bottom_masked_image = self.board.get_masked_bottom()
        dim = int(np.round(self.max_uv_range * 2 + 1))
        uv_luminance_mask = np.zeros((dim, dim))
        enclosed_FOV, local_FOV = self.get_FOV(self.fish.body.position, self.max_uv_range, self.env_variables["arena_width"],
                                               self.env_variables["arena_height"])
        lum_slice = self.board.global_luminance_mask[enclosed_FOV[0]:enclosed_FOV[1],
                                               enclosed_FOV[2]:enclosed_FOV[3]]
        uv_luminance_mask[local_FOV[0]:local_FOV[1],
                             local_FOV[2]:local_FOV[3]] = lum_slice
        uv_luminance_mask *= self.board.uv_scatter
        # Convert to FOV coordinates (to match eye coordinates)
        #full_masked_image=None
        if len(prey_locations) > 0:
            prey_locations_array = np.array(prey_locations) - np.array(self.fish.body.position) + self.max_uv_range
        else:
            prey_locations_array = np.array([])
        if len(sand_grain_locations) > 0:
            sand_grain_locations_array = np.array(sand_grain_locations) - np.array(
                self.fish.body.position) + self.board.max_visual_distance
        else:
            sand_grain_locations_array = np.empty((0, 2))

        # check if preditor exists
        if predator_bodies.size > 0:
            predator_left, predator_right, predator_distance = self.get_predator_angles_distance()
        else:
            predator_left, predator_right, predator_distance = np.nan, np.nan, np.nan

        self.fish.left_eye.read(bottom_masked_image, left_eye_pos[0], left_eye_pos[1], self.fish.body.angle, uv_luminance_mask,
                                prey_locations_array, sand_grain_locations_array, predator_left, predator_right, predator_distance)
        self.fish.right_eye.read(bottom_masked_image, right_eye_pos[0], right_eye_pos[1], self.fish.body.angle, uv_luminance_mask,
                                 prey_locations_array, sand_grain_locations_array, predator_left, predator_right, predator_distance)
        observation = np.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))
        return observation
