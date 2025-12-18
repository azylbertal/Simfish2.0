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
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as draw
from skimage import io
import h5py
from matplotlib.animation import FFMpegWriter
from sklearn.decomposition import PCA
from scipy.signal import detrend
from scipy.stats import zscore
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from simulation.define_actions import Actions
import cv2
from skimage.draw import line
from skimage import img_as_float

def draw_lines_with_opacity(image, x_coords, y_coords, colors, L, thickness=5):
    """
    Draw L lines on an image with smoothly changing opacity from 0 to 1.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (H, W, 3) float32 format with values 0-1
    x_coords : array-like
        Array of N x-coordinates
    y_coords : array-like  
        Array of N y-coordinates
    colors : array-like
        Array of L RGB colors, each color is [R, G, B] with values 0-1
    L : int
        Number of lines to draw (using the last L coordinate pairs)
    thickness : int
        Line thickness in pixels (default: 2)
    
    Returns:
    --------
    numpy.ndarray
        Image with overlayed lines (float32, 0-1 range)
    """
    
    # Work with a copy of the image
    result_image = image.copy()
    
    # Ensure we have enough coordinates
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    colors = np.array(colors, dtype=np.float32)
    
    if len(x_coords) < L + 1 or len(y_coords) < L + 1:
        raise ValueError(f"Need at least {L+1} coordinate points to draw {L} lines")
    
    if len(colors) != L:
        raise ValueError(f"Need exactly {L} colors for {L} lines")
    
    # Ensure image is 3D (convert grayscale to color if needed)
    if result_image.ndim == 2:
        result_image = np.stack([result_image] * 3, axis=-1)
    
    # Create single overlay and alpha mask
    overlay = np.zeros_like(result_image, dtype=np.float32)
    alpha_mask = np.zeros(result_image.shape[:2], dtype=np.float32)
    
    # Draw all lines on the single overlay
    start_idx = len(x_coords) - L - 1
    
    for i in range(L):
        # Calculate opacity (0 for oldest line, 1 for newest line)
        opacity = i / (L - 1) if L > 1 else 1.0
        
        if opacity == 0:
            continue  # Skip completely transparent lines
        
        # Get start and end coordinates for this line
        coord_idx = start_idx + i
        x1, y1 = int(x_coords[coord_idx]), int(y_coords[coord_idx])
        x2, y2 = int(x_coords[coord_idx + 1]), int(y_coords[coord_idx + 1])
        
        # Create temporary mask for this line (cv2 requires uint8 for line drawing)
        line_mask = np.zeros(result_image.shape[:2], dtype=np.uint8)
        

        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness)
        
        # Convert to boolean mask
        line_pixels = line_mask > 0
        
        # Update overlay with weighted color
        color_float = np.array([colors[i][0], colors[i][1], colors[i][2]], dtype=np.float32)
        
        # For overlapping pixels, use the maximum opacity (newest line wins)
        new_alpha = np.full_like(alpha_mask, opacity)
        update_mask = line_pixels & (new_alpha > alpha_mask)
        
        # Update overlay colors where this line should be visible
        for c in range(3):
            overlay[update_mask, c] = color_float[c]
        
        # Update alpha mask
        alpha_mask[update_mask] = opacity
    
    # Apply the blending using the alpha mask
    alpha_3d = np.stack([alpha_mask] * 3, axis=-1)
    result_image = result_image * (1 - alpha_3d) + overlay * alpha_3d
    
    return result_image

actions = Actions()
actions.from_hdf5('actions_all_bouts_with_null.h5')
actions = actions.get_all_actions()

class Vanishing_Line(object):
    def __init__(self, n_points, tail_length, rgb_color):
        self.n_points = int(n_points)
        self.tail_length = int(tail_length)
        self.rgb_color = rgb_color

    def set_data(self, x=None, y=None, z=None):
        if x is None or y is None or z is None:
            self.lc = Line3DCollection([])
        else:
            x = x[-self.n_points:]
            y = y[-self.n_points:]
            z = z[-self.n_points:]
            
            self.points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            
            self.segments = np.concatenate([self.points[:-1], self.points[1:]],
                                           axis=1)
            if hasattr(self, 'alphas'):
                del self.alphas
            if hasattr(self, 'rgba_colors'):
                del self.rgba_colors
            self.lc.set_segments(self.segments)
            self.lc.set_color(self.get_colors())
            self.lc.set_linewidth(2)

    def get_LineCollection(self):
        if not hasattr(self, 'lc'):
            self.set_data()
        return self.lc


    def get_alphas(self):
        n = len(self.points)
        if n < self.n_points:
            rest_length = self.n_points - self.tail_length
            if n <= rest_length:
                return np.ones(n)
            else:
                tail_length = n - rest_length
                tail = np.linspace(1./tail_length, 1., tail_length)
                rest = np.ones(rest_length)
                return np.concatenate((tail, rest))
        else: # n == self.n_points
            if not hasattr(self, 'alphas'):
                tail = np.linspace(1./self.tail_length, 1., self.tail_length)
                rest = np.ones(self.n_points - self.tail_length)
                self.alphas = np.concatenate((tail, rest))
            return self.alphas
        
    def get_colors(self):
        n = len(self.points)
        if  n < 2:
            return [self.rgb_color+[1.] for i in range(n)]
        if n < self.n_points:
            alphas = self.get_alphas()
            rgba_colors = np.zeros((n, 4))
            # first place the rgb color in the first three columns
            rgba_colors[:,0:3] = self.rgb_color
            # and the fourth column needs to be your alphas
            rgba_colors[:, 3] = alphas
            return rgba_colors
        else:
            if hasattr(self, 'rgba_colors'):
                pass
            else:
                alphas = self.get_alphas()
                rgba_colors = np.zeros((n, 4))
                # first place the rgb color in the first three columns
                rgba_colors[:,0:3] = self.rgb_color
                # and the fourth column needs to be your alphas
                rgba_colors[:, 3] = alphas
                self.rgba_colors = rgba_colors
            return self.rgba_colors
                
class DrawingBoard:

    def __init__(self, env_variables, data, include_background):
        self.width = env_variables["arena_width"]
        self.height = env_variables["arena_height"]
        self.light_gradient = env_variables["arena_light_gradient"]
        self.dark_gain = env_variables["arena_dark_gain"]
        self.dark_light_ratio = env_variables["arena_dark_fraction"]
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
        #self.apply_light()

    def get_base_arena(self, bkg=0.0):
        db = (np.ones((self.height, self.width, 3), dtype=np.double) * bkg)
        db[1:5, :] = np.array([1, 0, 0])
        db[self.width - 5:self.width - 1, :] = np.array([1, 0, 0])
        db[:, 1:5] = np.array([1, 0, 0])
        db[:, self.height - 5:self.height - 1] = np.array([1, 0, 0])
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

    def show_action_discrete(self, fish_angle, x_position, y_position, colour, name=None):
        if name:
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (int(x_position), int(y_position))
            fontScale              = 0.5
            fontColor              = colour
            thickness              = 3
            lineType               = 2

            cv2.putText(self.db,name, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            
        else:
            rr, cc = draw.ellipse(int(y_position), int(x_position), 5, 3, rotation=-fish_angle)
            self.db[rr, cc, :] = colour

    def show_consumption(self, fish_angle, x_position, y_position, colour):
        rr, cc = draw.ellipse(int(y_position), int(x_position), 10, 6, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def get_action_colour(self, action):
        """Returns the (R, G, B) for associated actions"""
        return actions[action]['color']
        # hex_col = actions[action]['color']
        # hex_col = hex_col.lstrip('#')
        # rgb = tuple(int(hex_col[i:i+2], 16)/255 for i in (0, 2, 4))
        # return rgb 

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
            




def draw_previous_actions(board, past_actions, past_positions,
                          n_actions_to_show):
    if len(past_actions) > 2:
        L=min(n_actions_to_show, len(past_positions)-1)
        colors = [board.get_action_colour(a) for a in past_actions[-L:]]
        board.db = draw_lines_with_opacity(board.db,
                                            [pos[0] for pos in past_positions],
                                            [pos[1] for pos in past_positions],
                                            colors,
                                            L=L)
    # while len(past_actions) > n_actions_to_show:
    #     past_actions.pop(0)
    # while len(past_positions) > n_actions_to_show:
    #     past_positions.pop(0)
    # while len(fish_angles) > n_actions_to_show:
    #     fish_angles.pop(0)
    # while len(consumption_buffer) > n_actions_to_show:
    #     consumption_buffer.pop(0)

    # for i, a in enumerate(past_actions):
    #     if continuous_actions:
    #         if a[1] < 0:
    #             action_colour = (
    #             adjusted_colour_index, bkg_scatter, bkg_scatter)
    #         else:
    #             action_colour = (bkg_scatter, adjusted_colour_index, adjusted_colour_index)

    #         board.show_action_continuous(a[0], a[1], fish_angles[i], past_positions[i][0],
    #                                           past_positions[i][1], action_colour)
    #     else:
    #         action_colour = board.get_action_colour(past_actions[i])
    #         action_name = actions[past_actions[i]]['name']
    #         # if '_' in string, only take what's before it
    #         if '_' in action_name:
    #             action_name = action_name.split('_')[0]
    #         board.show_action_discrete(fish_angles[i], past_positions[i][0],
    #                                         past_positions[i][1], action_colour, name=None)
        # if consumption_buffer is not None:
        #     if consumption_buffer[i] == 1:
        #         board.show_consumption(fish_angles[i], past_positions[i][0],
        #                                past_positions[i][1], (1, 0, 0))
    return board, past_actions, past_positions




def draw_episode(data_file, video_file, continuous_actions=False,  draw_past_actions=True, show_energy_state=False,
                 trim_to_fish=False, showed_region_quad=500, n_actions_to_show=500,
                 include_background=False, num_steps=1000):
    

    with h5py.File(data_file, 'r') as datfl:
        env_variables = dict(datfl['env_variables'].attrs)
        data = {}
        for key in datfl.keys():
            data[key] = np.array(datfl[key])
    if 'actor_state' in data:
        rnn_states = data['actor_state']
        rnn_states = rnn_states[:, np.std(rnn_states, axis=0) > 1e-6]
        rnn_states = detrend(zscore(rnn_states), axis=0)
        with_states = True
    else:
        rnn_states = np.zeros((data['fish_x'].shape[0], 3))
        with_states = False
    print(f"Used RNN dimensions: {rnn_states.shape[1]}")
    rnn_states_PCA = PCA(n_components=3).fit_transform(rnn_states)
    fig = plt.figure(facecolor='0.0', figsize=(15, 12.69), dpi=100)
    gs = fig.add_gridspec(nrows=11, ncols=13, left=0.05, right=0.85,
                      hspace=0.1, wspace=0.1)
    ax0 = fig.add_subplot(gs[0:10, 0:10])
    
    ax1 = fig.add_subplot(gs[10, 0:4])
    ax2 = fig.add_subplot(gs[10, 6:10])
    # ax3 = fig.add_subplot(gs[0, 6:])
    ax3 = fig.add_subplot(gs[5:8, 10:13], projection='3d') # pca in 3d
    ax3.set_facecolor((0,0,0))
    ax4 = fig.add_subplot(gs[10, 10:13])
    ax4.set_facecolor((0,0,0))
    ax5 = fig.add_subplot(gs[0:4, 10:12])
    ax5.set_facecolor((0,0,0))
    # draw short lines representing all actions (in appropriate color), with action names as text labels
    hh = 0
    for i in range(len(actions)):
        if not '_L' in actions[i]['name']:
            
            ax5.plot([0, 1], [hh, hh], color=actions[i]['color'], linewidth=3)
            action_name = actions[i]['name'].split('_')[0]
            ax5.text(1.2, hh-0.1, action_name, fontsize=12, color=actions[i]['color'])
            hh += 1
    # remove all ticks and spines
    ax5.set_xticks([])
    ax5.set_yticks([])
    for spine in ax5.spines.values():
        spine.set_visible(False)
    ax5.set_ylim(-1, hh+1)
    ax5.set_xlim(0, 5)
    
        # ax6 = fig.add_subplot(gs[3, 6:])
    # ax7 = fig.add_subplot(gs[4, 6:])
    #annotate_axes(ax1, 'ax1')

    board = DrawingBoard(env_variables, data, include_background)
    energy_levels = data["internal_state"][:, 1]
    salt_level = data["internal_state"][:, 2]
    fish_positions = np.array([data['fish_x'], data['fish_y']]).T
    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    print(writer.supported_formats)
    action_buffer = []
    position_buffer = []
    orientation_buffer = []
    consumption_buffer = []
    if len(fish_positions) < num_steps: # Don't try to draw more steps than exist
        num_steps = len(fish_positions)

        #frames = np.zeros((num_steps, int(env_variables["arena_height"]*scale), int((env_variables["arena_width"]+addon)*scale), 3))
    with writer.saving(fig, video_file, 100):
        for step in range(num_steps):
            board.db = board.get_base_arena(0.3)

            print(f"{step}/{num_steps}")
            if continuous_actions:
                action_buffer.append([data["impulse"][step], data["angle"][step]])
            else:
                action_buffer.append(data["action"][step])
            position_buffer.append(fish_positions[step])
            orientation_buffer.append(data["fish_angle"][step])
            #consumption_buffer.append(data["consumed"][step])




            if show_energy_state:
                fish_body_colour = (1-energy_levels[step], energy_levels[step], 0)
            else:
                fish_body_colour = (0, 1, 0)


            # Draw prey
            px = np.round(data['prey_x'][step, :]).astype(int)
            py = np.round(data['prey_y'][step, :]).astype(int)
            nan_prey = np.isnan(px)
            px = px[~nan_prey]
            py = py[~nan_prey]
            rrs, ccs = board.multi_circles(px, py, 7)#env_variables["prey_size_visualisation"])

            rrs = np.clip(rrs, 0, env_variables["arena_width"]-1)
            ccs = np.clip(ccs, 0, env_variables["arena_height"]-1)

            board.db[rrs, ccs] = (1, 0.3, 1.0)
            board.apply_light()
            if draw_past_actions:
                board, action_buffer, position_buffer = draw_previous_actions(board, action_buffer,
                                                                                                position_buffer, n_actions_to_show=n_actions_to_show)
            board.fish_shape(fish_positions[step], env_variables['fish_mouth_radius']*2,
                                env_variables['fish_head_radius']*2, env_variables['fish_tail_length']*2,
                            (0.8, 0.8, 1), (0.7,0.7,1), data["fish_angle"][step])

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
            left_obs = np.transpose(np.tile(left_obs, [3, 1, 1]), [1, 2, 0]) / 32
            left_obs[0,:,1] = 0
            left_obs[1:,:,2] = 0
            
            right_obs = data['vis_observation'][step, :, :, 1].T
            right_obs = np.transpose(np.tile(right_obs, [3, 1, 1]), [1, 2, 0]) / 32
            right_obs[0,:,1] = 0
            right_obs[1:,:,2] = 0
            ax1.clear()
            ax2.clear()
            ax1.imshow(left_obs, interpolation='nearest', aspect='auto', vmin=0, vmax=64)
            ax1.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax2.imshow(right_obs, interpolation='nearest', aspect='auto', vmin=0, vmax=64)
            ax2.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

            plot_start = max(0, step - 200)
            # ax4.clear()
            # ax4.plot(rnn_states_PCA[plot_start:step, :], linewidth=0.5)
            # ax4.tick_params(left = False, right = False , labelleft = False ,
            #         labelbottom = False, bottom = False)
            ax3.clear()
            rgb_color = [0.2, 0.2, 1.0]
            line = Vanishing_Line(50, 25, rgb_color)
            line.set_data()
            line.set_data(rnn_states_PCA[:step,0], rnn_states_PCA[:step,1], rnn_states_PCA[:step,2])
            ax3.add_collection(line.get_LineCollection())
            ax3.scatter(rnn_states_PCA[step-1,0], rnn_states_PCA[step-1,1], rnn_states_PCA[step-1,2], color='blue', s=5)
            ax3.set_xlim((np.percentile(rnn_states_PCA[:,0], 5), np.percentile(rnn_states_PCA[:,0], 95)))
            ax3.set_ylim((np.percentile(rnn_states_PCA[:,1], 5), np.percentile(rnn_states_PCA[:,1], 95)))
            ax3.set_zlim((np.percentile(rnn_states_PCA[:,2], 5), np.percentile(rnn_states_PCA[:,2], 95)))
            ax3.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

            ax4.clear()
            ax4.plot(energy_levels[plot_start:step], linewidth=2, color=[1,0,0])
            ax4.plot(salt_level[plot_start:step], linewidth=2, color='cyan')
            ax4.set_ylim((0,1.1))
            ax4.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            writer.grab_frame()
    writer.finish()


if __name__ == "__main__":

    data_file = str(sys.argv[1])
    video_file = str(sys.argv[2])
    # data_file = '/home/asaph/cs_cluster/Simfish2.0/test_stage1/logs/evaluator/logs_70.hdf5'
    num_steps = int(sys.argv[3]) if len(sys.argv)>3 else 1000
    draw_episode(data_file, video_file, continuous_actions=False, show_energy_state=False, draw_past_actions=True,
                 trim_to_fish=True, showed_region_quad=600, include_background=True, n_actions_to_show=25, num_steps=num_steps)




