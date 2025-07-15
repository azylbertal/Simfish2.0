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
import json
import matplotlib.pyplot as plt
import skimage.draw as draw
from skimage import io
import h5py
from skimage.transform import resize, rescale
from matplotlib.animation import FFMpegWriter


class DrawingBoard:

    def __init__(self, env_variables, data, include_background):
        self.width = env_variables["arena_width"]
        self.height = env_variables["arena_height"]
        self.light_gradient = env_variables["light_gradient"]
        self.dark_gain = env_variables["dark_gain"]
        self.dark_light_ratio = env_variables["dark_light_ratio"]
        if env_variables["test_sensory_system"]:
            self.dark_light_ratio = 0.5
        self.include_background = include_background
        if env_variables["salt"]:
            self.salt_location = data["salt_location"][0]
        else:
            self.salt_location = [np.nan, np.nan]
        if include_background:
            self.background = data["sediment"][0,:, :]
            self.background = np.expand_dims(self.background/10, 2)
            self.background = np.concatenate((self.background,
                                             self.background,
                                             np.zeros(self.background.shape)), axis=2)

        self.db = self.get_base_arena(0.3)
        self.apply_light()

    def get_base_arena(self, bkg=0.0):
        db = (np.ones((self.height, self.width, 3), dtype=np.double) * bkg)
        db[1:2, :] = np.array([1, 0, 0])
        db[self.width - 2:self.width - 1, :] = np.array([1, 0, 0])
        db[:, 1:2] = np.array([1, 0, 0])
        db[:, self.height - 2:self.height - 1] = np.array([1, 0, 0])
        if self.include_background:
            db += self.background*0.05
        return db

    def circle(self, center, rad, color):
        rr, cc = draw.disk((center[1], center[0]), rad, shape=self.db.shape)
        self.db[rr, cc, :] = color

    @staticmethod
    def multi_circles(cx, cy, rad):
        rr, cc = draw.disk((0, 0), rad)
        rrs = np.tile(rr, (len(cy), 1)) + np.tile(np.reshape(cy, (len(cy), 1)), (1, len(rr)))
        ccs = np.tile(cc, (len(cx), 1)) + np.tile(np.reshape(cx, (len(cx), 1)), (1, len(cc)))
        return rrs, ccs

    def overlay_salt(self):
        """Show salt source"""
        # Consider modifying so that shows distribution.
        if np.isnan(self.salt_location[0]) or np.isnan(self.salt_location[1]):
            return
        self.circle(self.salt_location, 20, (0, 1, 1))

    def tail(self, head, left, right, tip, color):
        tail_coordinates = np.array((head, left, tip, right))
        rr, cc = draw.polygon(tail_coordinates[:, 1], tail_coordinates[:, 0], self.db.shape)
        self.db[rr, cc, :] = color

    def fish_shape(self, mouth_centre, mouth_rad, head_rad, tail_length, mouth_colour, body_colour, angle):
        offset = np.pi / 2
        angle += offset
        angle = -angle
        self.circle(mouth_centre, mouth_rad, mouth_colour)  # For the mouth.
        dx1, dy1 = head_rad * np.sin(angle), head_rad * np.cos(angle)
        head_centre = (mouth_centre[0] + dx1,
                       mouth_centre[1] + dy1)
        self.circle(head_centre, head_rad, body_colour)
        dx2, dy2 = -1 * dy1, dx1
        left_flank = (head_centre[0] + dx2,
                      head_centre[1] + dy2)
        right_flank = (head_centre[0] - dx2,
                       head_centre[1] - dy2)
        tip = (mouth_centre[0] + (tail_length + head_rad) * np.sin(angle),
               mouth_centre[1] + (tail_length + head_rad) * np.cos(angle))
        self.tail(head_centre, left_flank, right_flank, tip, body_colour)

    def show_action_continuous(self, impulse, angle, fish_angle, x_position, y_position, colour):
        # rr, cc = draw.ellipse(int(y_position), int(x_position), (abs(angle) * 3) + 3, (impulse*0.5) + 3, rotation=-fish_angle)
        rr, cc = draw.ellipse(int(y_position), int(x_position), 3, (impulse*0.5) + 3, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def show_action_discrete(self, fish_angle, x_position, y_position, colour):
        rr, cc = draw.ellipse(int(y_position), int(x_position), 5, 3, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def show_consumption(self, fish_angle, x_position, y_position, colour):
        rr, cc = draw.ellipse(int(y_position), int(x_position), 10, 6, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def get_action_colour(self, action):
        """Returns the (R, G, B) for associated actions"""
        if action == 0:  # Slow2
            action_colour = (0, 1, 0)

        elif action == 1:  # RT right
            action_colour = (0, 1, 0)

        elif action == 2:  # RT left
            action_colour = (0, 1, 0)

        elif action == 3:  # Short capture swim
            action_colour = (1, 0, 1)

        elif action == 4:  # j turn right
            action_colour = (1, 1, 1)

        elif action == 5:  # j turn left
            action_colour = (1, 1, 1)

        elif action == 6:  # Do nothing
            action_colour = (0, 0, 0)

        elif action == 7:  # c start right
            action_colour = (1, 0, 0)

        elif action == 8:  # c start left
            action_colour = (1, 0, 0)

        elif action == 9:  # Approach swim.
            action_colour = (0, 1, 0)

        elif action == 10:  # j turn right (large)
            action_colour = (1, 1, 1)

        elif action == 11:  # j turn left (large)
            action_colour = (1, 1, 1)

        else:
            action_colour = (0, 0, 0)
            print("Invalid action given")

        return action_colour

    def apply_light(self):
        dark_field_length = int(self.height * self.dark_light_ratio)
        if self.light_gradient > 0 and dark_field_length > 0:
            self.db[:int(dark_field_length - (self.light_gradient / 2)), :] *= np.sqrt(self.dark_gain)

            gradient = np.linspace(np.sqrt(self.dark_gain), 1, self.light_gradient)
            gradient = np.expand_dims(gradient, 1)
            gradient = np.repeat(gradient, self.height, 1)
            #gradient = self.chosen_math_library.expand_dims(gradient, 2)
            self.db[int(dark_field_length - (self.light_gradient / 2)):int(dark_field_length + (self.light_gradient / 2)), :, 0] *= gradient
            self.db[int(dark_field_length - (self.light_gradient / 2)):int(dark_field_length + (self.light_gradient / 2)), :, 1] *= gradient
            self.db[int(dark_field_length - (self.light_gradient / 2)):int(dark_field_length + (self.light_gradient / 2)), :, 2] *= gradient

        else:
            self.db[:dark_field_length, :] *= np.sqrt(self.dark_gain)
            




def draw_previous_actions(board, past_actions, past_positions, fish_angles, adjusted_colour_index,
                          continuous_actions, n_actions_to_show, bkg_scatter, consumption_buffer=None):
    while len(past_actions) > n_actions_to_show:
        past_actions.pop(0)
    while len(past_positions) > n_actions_to_show:
        past_positions.pop(0)
    while len(fish_angles) > n_actions_to_show:
        fish_angles.pop(0)
    while len(consumption_buffer) > n_actions_to_show:
        consumption_buffer.pop(0)

    for i, a in enumerate(past_actions):
        if continuous_actions:
            if a[1] < 0:
                action_colour = (
                adjusted_colour_index, bkg_scatter, bkg_scatter)
            else:
                action_colour = (bkg_scatter, adjusted_colour_index, adjusted_colour_index)

            board.show_action_continuous(a[0], a[1], fish_angles[i], past_positions[i][0],
                                              past_positions[i][1], action_colour)
        else:
            action_colour = board.get_action_colour(past_actions[i])
            board.show_action_discrete(fish_angles[i], past_positions[i][0],
                                            past_positions[i][1], action_colour)
        # if consumption_buffer is not None:
        #     if consumption_buffer[i] == 1:
        #         board.show_consumption(fish_angles[i], past_positions[i][0],
        #                                past_positions[i][1], (1, 0, 0))
    return board, past_actions, past_positions, fish_angles




def draw_episode(data_file, config_file, continuous_actions=False,  draw_past_actions=True, show_energy_state=False,
                 scale=1.0, trim_to_fish=False, showed_region_quad=500, n_actions_to_show=500,
                 save_id="placeholder", s_per_frame=0.03, include_background=False):
    
    with open(config_file, 'r') as f:
        env_variables = json.load(f)
    with h5py.File(data_file, 'r') as datfl:
        data = {}
        for key in datfl.keys():
            data[key] = np.array(datfl[key])
    fig = plt.figure(facecolor='0.9', figsize=(14, 14), dpi=100)
    gs = fig.add_gridspec(nrows=9, ncols=9, left=0.05, right=0.85,
                      hspace=0.1, wspace=0.1)
    ax0 = fig.add_subplot(gs[:-1, 0:8])
    
    ax1 = fig.add_subplot(gs[-1, 0:4])
    ax2 = fig.add_subplot(gs[-1, 4:8])
    # ax3 = fig.add_subplot(gs[0, 6:])
    # ax4 = fig.add_subplot(gs[1, 6:])
    # ax5 = fig.add_subplot(gs[2, 6:])
    # ax6 = fig.add_subplot(gs[3, 6:])
    # ax7 = fig.add_subplot(gs[4, 6:])
    #annotate_axes(ax1, 'ax1')

    board = DrawingBoard(env_variables, data, include_background)
    if show_energy_state:
        energy_levels = data["energy_state"]
    fish_positions = np.array([data['fish_x'], data['fish_y']]).T
    num_steps = fish_positions.shape[0]
    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    print(writer.supported_formats)
    action_buffer = []
    position_buffer = []
    orientation_buffer = []
    consumption_buffer = []
                
        #frames = np.zeros((num_steps, int(env_variables["arena_height"]*scale), int((env_variables["arena_width"]+addon)*scale), 3))
    with writer.saving(fig, "writer_test.mp4", 100):
        for step in range(num_steps):
            print(f"{step}/{num_steps}")
            if continuous_actions:
                action_buffer.append([data["impulse"][step], data["angle"][step]])
            else:
                action_buffer.append(data["action"][step])
            position_buffer.append(fish_positions[step])
            orientation_buffer.append(data["fish_angle"][step])
            #consumption_buffer.append(data["consumed"][step])

            if draw_past_actions:
                # adjusted_colour_index = ((1 - env_variables["bkg_scatter"]) * (step + 1) / n_actions_to_show) + \
                #                         env_variables["bkg_scatter"]
                # adjusted_colour_index = (1 - env_variables["bkg_scatter"]) + env_variables["bkg_scatter"]
                adjusted_colour_index = 1
                board, action_buffer, position_buffer, orientation_buffer = draw_previous_actions(board, action_buffer,
                                                                                                position_buffer, orientation_buffer,
                                                                                                adjusted_colour_index=adjusted_colour_index,
                                                                                                continuous_actions=continuous_actions,
                                                                                                n_actions_to_show=n_actions_to_show,
                                                                                                bkg_scatter=env_variables["background_brightness"],
                                                                                                consumption_buffer=consumption_buffer)



            if show_energy_state:
                fish_body_colour = (1-energy_levels[step], energy_levels[step], 0)
            else:
                fish_body_colour = (0, 1, 0)

            board.fish_shape(fish_positions[step], env_variables['fish_mouth_radius'],
                                env_variables['fish_head_radius'], env_variables['fish_tail_length'],
                            (0, 1, 0), fish_body_colour, data["fish_angle"][step])

            # Draw prey
            # px = np.round(np.array([pr[0] for pr in data["prey_positions"][step]])).astype(int)
            # py = np.round(np.array([pr[1] for pr in data["prey_positions"][step]])).astype(int)
            px = np.round(data['prey_x'][step, :]).astype(int)
            py = np.round(data['prey_y'][step, :]).astype(int)
            nan_prey = np.isnan(px)
            px = px[~nan_prey]
            py = py[~nan_prey]
            rrs, ccs = board.multi_circles(px, py, 10)#env_variables["prey_size_visualisation"])

            rrs = np.clip(rrs, 0, env_variables["arena_width"]-1)
            ccs = np.clip(ccs, 0, env_variables["arena_height"]-1)

            board.db[rrs, ccs] = (0.5, 0.5, 1)

            # Draw sand grains
            if env_variables["sand_grain_num"] > 0:
                px = np.round(np.array([pr.position[0] for pr in data["sand_grain_positions"]])).astype(int)
                py = np.round(np.array([pr.position[1] for pr in data["sand_grain_positions"]])).astype(int)
                rrs, ccs = board.multi_circles(px, py, env_variables["prey_size_visualisation"])

                rrs = np.clip(rrs, 0, env_variables["arena_width"] - 1)
                ccs = np.clip(ccs, 0, env_variables["arena_height"] - 1)

                board.db_visualisation[rrs, ccs] = (0, 0, 1)

            if data["predator_x"][step]!=0 and data["predator_y"][step]!=0:
                predator_position = (data["predator_x"][step], data["predator_y"][step])
                board.circle(predator_position, env_variables['predator_radius'], (0, 1, 0))
            board.overlay_salt()
            frame = board.db

            if trim_to_fish:
                centre_y, centre_x = fish_positions[step][0], fish_positions[step][1]
                # print(centre_x, centre_y)
                dist_x1 = centre_x
                dist_x2 = env_variables["arena_width"] - centre_x
                dist_y1 = centre_y
                dist_y2 = env_variables["arena_height"] - centre_y
                # print(dist_x1, dist_x2, dist_y1, dist_y2)
                if dist_x1 < showed_region_quad:
                    centre_x += showed_region_quad - dist_x1
                elif dist_x2 < showed_region_quad:
                    centre_x -= showed_region_quad - dist_x2
                if dist_y1 < showed_region_quad:
                    centre_y += showed_region_quad - dist_y1
                if dist_y2 < showed_region_quad:
                    centre_y -= showed_region_quad - dist_y2
                centre_x = int(centre_x)
                centre_y = int(centre_y)
                # Compute centre position - so can deal with edges
                frame = frame[centre_x-showed_region_quad:centre_x+showed_region_quad,
                        centre_y-showed_region_quad:centre_y+showed_region_quad]

            this_frame = frame#rescale(frame, scale, channel_axis=2, anti_aliasing=True)
            ax0.clear()
            ax0.imshow(this_frame, interpolation='nearest', aspect='auto')
            ax0.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

            left_obs = data['vis_observation'][step, :, :, 0].T

            right_obs = data['vis_observation'][step, :, :, 1].T
            ax1.clear()
            ax2.clear()
            ax1.imshow(left_obs, interpolation='nearest', aspect='auto', vmin=0, vmax=64)
            ax1.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax2.imshow(right_obs, interpolation='nearest', aspect='auto', vmin=0, vmax=64)
            ax2.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

            # plot_start = max(0, step - 100)
            # ax3.clear()
            # ax3.plot(energy_levels[plot_start:step], color='green')
            # ax3.tick_params(left = False, right = False , labelleft = False ,
            #         labelbottom = False, bottom = False)
            # ax4.clear()
            # ax4.plot(data['rnn_state_actor'][plot_start:step, 0, :10])
            # ax4.tick_params(left = False, right = False , labelleft = False ,
            #         labelbottom = False, bottom = False)
            # ax5.clear()
            # ax5.plot(data['rnn_state_actor'][plot_start:step, 0, 10:20])
            # ax5.tick_params(left = False, right = False , labelleft = False ,
            #         labelbottom = False, bottom = False)
            # ax6.clear()
            # ax6.plot(data['rnn_state_actor'][plot_start:step, 0, 20:30])
            # ax6.tick_params(left = False, right = False , labelleft = False ,
            #         labelbottom = False, bottom = False)
            # ax7.clear()
            # ax7.plot(data['rnn_state_actor'][plot_start:step, 0, 30:40])
            # ax7.tick_params(left=False, right=False , labelleft=False, labelbottom=False, bottom=False)
            board.db = board.get_base_arena(0.3)
            board.apply_light()
            writer.grab_frame()
    writer.finish()


    #make_video(frames, 'try.mp4', duration=len(frames) * s_per_frame, true_image=True)


if __name__ == "__main__":
    model = "local_test_large"

    config_file = './env_config/test_env.json'
    data_file = '/home/asaph/acme/20250715-160023/logs/evaluator/logs_1.hdf5'

    draw_episode(data_file, config_file, continuous_actions=False, show_energy_state=False,
                 trim_to_fish=True, showed_region_quad=600, save_id="ep1902", include_background=True, n_actions_to_show=10)




