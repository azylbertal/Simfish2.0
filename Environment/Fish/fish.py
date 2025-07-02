import numpy as np
import pymunk
#import cupy as cp

from Environment.Fish.eye import Eye


def draw_angle_dist_bak(bout_id):
    if bout_id == 8:  # Slow2
        mean = [2.49320953e+00, 2.36217665e-19]
        cov = [[4.24434912e-01, 1.89175382e-18],
                [1.89175382e-18, 4.22367139e-03]]
    elif bout_id == 7:  # RT
        mean = [2.74619216, 0.82713249]
        cov = [[0.3839484,  0.02302918],
               [0.02302918, 0.03937928]]
    elif bout_id == 0:  # sCS
        mean = [0.956603146, -6.86735892e-18]
        cov = [[2.27928786e-02, 1.52739195e-19],
               [1.52739195e-19, 3.09720798e-03]]
    elif bout_id == 4:  # J-turn 1
        mean = [0.49074911, 0.39750791]
        cov = [[0.00679925, 0.00071446],
               [0.00071446, 0.00626601]]
    elif bout_id == 44:  # J-turn 2
        mean = [1.0535197,  0.61945679]
        # cov = [[ 0.0404599,  -0.00318193],
        #        [-0.00318193,  0.01365224]]
        cov = [[0.0404599,  0.0],
               [0.0,  0.01365224]]
    elif bout_id == 5:  # C-Start
        mean = [7.03322223, 0.67517832]
        cov = [[1.35791922, 0.10690938],
               [0.10690938, 0.10053853]]
    elif bout_id == 10:  # AS
        mean = [6.42048088e-01, 1.66490488e-17]
        cov = [[3.99909515e-02, 3.58321400e-19],
               [3.58321400e-19, 3.24366068e-03]]
    else:
        mean = [0, 0],
        cov = [[0, 0],
               [0, 0]]
        print("Draw action error")

    bout_vals = np.random.multivariate_normal(mean, cov, 1)
    return bout_vals[0, 1], bout_vals[0, 0], mean[1], mean[0]

def draw_angle_dist(bout_id):
    if bout_id == 0:  # Slow2
        mean = [2.49320953e+00, 2.36217665e-19]
        cov = [[4.24434912e-01, 1.89175382e-18],
                [1.89175382e-18, 4.22367139e-03]]
    elif bout_id == 1 or bout_id == 2:  # RT
        mean = [2.74619216, 0.82713249]
        cov = [[0.3839484,  0.02302918],
               [0.02302918, 0.03937928]]
    elif bout_id == 3:  # sCS
        mean = [0.956603146, -6.86735892e-18]
        cov = [[2.27928786e-02, 1.52739195e-19],
               [1.52739195e-19, 3.09720798e-03]]
    elif bout_id == 4 or bout_id == 5:  # J-turn 1
        mean = [0.49074911, 0.39750791]
        cov = [[0.00679925, 0.00071446],
               [0.00071446, 0.00626601]]
    elif bout_id == 6:  # do nothing
        mean = [0, 0]
        cov = [[0, 0],
               [0, 0]]
    elif bout_id == 7 or bout_id == 8:  # C-Start
        mean = [7.03322223, 0.67517832]
        cov = [[1.35791922, 0.10690938],
               [0.10690938, 0.10053853]]
    elif bout_id == 9:  # AS
        mean = [6.42048088e-01, 1.66490488e-17]
        cov = [[3.99909515e-02, 3.58321400e-19],
               [3.58321400e-19, 3.24366068e-03]]

    elif bout_id == 10 or bout_id == 11:  # J-turn 2
        mean = [1.0535197,  0.61945679]
        # cov = [[ 0.0404599,  -0.00318193],
        #        [-0.00318193,  0.01365224]]
        cov = [[0.0404599,  0.0],
               [0.0,  0.01365224]]
    else:
        mean = [0, 0],
        cov = [[0, 0],
               [0, 0]]
        print("Draw action error")

    bout_vals = np.random.multivariate_normal(mean, cov, 1)
    return bout_vals[0, 1], bout_vals[0, 0], mean[1], mean[0]

class Fish:
    """
    Created to simplify the SimState class, while making it easier to have environments with multiple agents in future.
    """

    def __init__(self, board, env_variables, max_uv_range):

        # For the purpose of producing a calibration curve.
        inertia = pymunk.moment_for_circle(env_variables['fish_mass'], 0, env_variables['fish_head_radius'], (0, 0))
        self.max_uv_range = max_uv_range
        self.env_variables = env_variables
        self.body = pymunk.Body(1, inertia)

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
        self.retinal_field = env_variables['visual_field'] * (np.pi / 180)
        self.conv_state = 0


        self.left_eye = Eye(board, self.verg_angle, self.retinal_field, True, env_variables,
                            max_uv_range=self.max_uv_range)
        self.right_eye = Eye(board, self.verg_angle, self.retinal_field, False, env_variables,
                             max_uv_range=self.max_uv_range)

        self.hungry = 0
        self.stress = 1
        self.prey_consumed = False
        self.touched_edge = False
        self.touched_predator = False
        self.making_capture = False
        self.capture_possible = False
        self.prev_action_impulse = 0
        self.prev_action_angle = 0

        # Energy system (new simulation)
        self.energy_level = 1.0
        self.i_scaling_energy_cost = self.env_variables['i_scaling_energy_cost']
        self.a_scaling_energy_cost = self.env_variables['a_scaling_energy_cost']
        self.baseline_energy_use = self.env_variables['baseline_energy_use']

        self.action_reward_scaling = self.env_variables['action_reward_scaling']
        self.consumption_reward_scaling = self.env_variables['consumption_reward_scaling']

        if "action_energy_use_scaling" in self.env_variables:
            self.action_energy_use_scaling = self.env_variables["action_energy_use_scaling"]
        else:
            self.action_energy_use_scaling = "Sublinear"

        # Salt health (new simulation)
        self.salt_health = 1.0

        # Touch edge - for penalty
        self.touched_edge_this_step = False

        self.impulse_vector_x = 0
        self.impulse_vector_y = 0

        self.deterministic_action = self.env_variables['deterministic_action']

    def take_action(self, action):
        reward = 0
        if self.env_variables['test_sensory_system']:
            self.body.angle += 0.1
        if not (action == 6 or self.env_variables['test_sensory_system']):
            angle_change, distance, mean_angle, mean_distance = draw_angle_dist(action)
            if self.deterministic_action:
                angle_change = mean_angle
                distance = mean_distance

            if action in [2, 5, 8, 11]:
                angle_change = -angle_change

            if action == 3:
                reward -= self.env_variables['capture_swim_extra_cost']
                self.making_capture = True

            self.prev_action_angle = angle_change
            self.body.angle += self.prev_action_angle
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
        else:
            self.prev_action_impulse = 0
            self.prev_action_angle = 0
            self.prev_action = action
            reward = 0



        if not action in range(0, 12):
            reward = None
            print("Invalid action given")

        return reward


    def calculate_impulse(self, distance):
        """
        Uses the derived distance-mass-impulse relationship to convert an input distance (in mm) to impulse
        (arbitrary units).
        :param distance:
        :return:
        """
        # return (distance * 10 - (0.004644 * self.env_variables['fish_mass'] + 0.081417)) / 1.771548
        # return (distance * 10) * 0.360574383  # From mm
        return (distance * 10) * 0.34452532909386484  # From mm

    def readings_to_photons(self, readings):
        """Rounds down observations to form array of discrete photon events."""
        photons = np.floor(readings).astype(int)
        photons = photons.clip(0, 255)

        return photons

    def update_energy_level(self, reward, consumption):
        """Updates the current energy state for continuous and discrete fish."""
        energy_intake = 1.0 * consumption

        if self.action_energy_use_scaling == "Nonlinear":
            energy_use = self.i_scaling_energy_cost * (abs(self.prev_action_impulse) ** 2) + \
                         self.a_scaling_energy_cost * (abs(self.prev_action_angle) ** 2) + \
                         self.baseline_energy_use
        elif self.action_energy_use_scaling == "Linear":
            energy_use = self.i_scaling_energy_cost * (abs(self.prev_action_impulse)) + \
                         self.a_scaling_energy_cost * (abs(self.prev_action_angle)) + \
                         self.baseline_energy_use
        elif self.action_energy_use_scaling == "Sublinear":
            energy_use = self.i_scaling_energy_cost * (abs(self.prev_action_impulse) ** 0.5) + \
                         self.a_scaling_energy_cost * (abs(self.prev_action_angle) ** 0.5) + \
                         self.baseline_energy_use
        else:
            energy_use = self.i_scaling_energy_cost * (abs(self.prev_action_impulse) ** 0.5) + \
                         self.a_scaling_energy_cost * (abs(self.prev_action_angle) ** 0.5) + \
                         self.baseline_energy_use

        # print()
        # print(self.prev_action_impulse)
        # print()
        if prev_action == 3:
            energy_use *= self.env_variables['capture_swim_energy_cost_scaling']
        reward += (energy_intake * self.consumption_reward_scaling) - (energy_use * self.action_reward_scaling)

        self.energy_level += energy_intake - energy_use
        if self.energy_level > 1.0:
            self.energy_level = 1.0

        return reward
