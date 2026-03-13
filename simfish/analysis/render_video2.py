import sys
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import detrend
from scipy.stats import zscore
from simulation import Actions

# --- Setup Actions ---
actions_handler = Actions()
actions_handler.from_hdf5('actions_all_bouts_with_null.h5')
actions_info = actions_handler.get_all_actions()

def get_bgr_from_action(color_data):
    if isinstance(color_data, str):
        hex_col = color_data.lstrip('#')
        rgb = tuple(int(hex_col[i:i+2], 16)/255.0 for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])
    return (float(color_data[2]), float(color_data[1]), float(color_data[0]))

class FastRenderer:
    def __init__(self, env_vars, data, include_background, quad):
        self.h, self.w = int(env_vars["arena_height"]), int(env_vars["arena_width"])
        self.quad = quad
        
        # 1. Start with the base background (0.3 matches render_video.py)
        # Using float32 for high-precision blending
        self.base_arena = np.full((self.h, self.w, 3), 0.3, dtype=np.float32)
        
        # 2. Add Sediment Background correctly
        if include_background and "sediment" in data:
            # Original script logic: sediment / 10 added to R and G channels
            sed = np.array(data["sediment"][0], dtype=np.float32) / 10.0
            
            # Ensure the sediment array matches arena dimensions
            # Add to Red and Green to create the 'yellowish' organic look
            self.base_arena[:, :, 0] += (sed * 0.05) 
            self.base_arena[:, :, 1] += (sed * 0.05)
        # 3. Apply the Light/Dark Field
        dark_fraction = env_vars.get("arena_dark_fraction", 0)
        dark_gain = env_vars.get("arena_dark_gain", 1.0)
        dark_field_limit = int(self.h * dark_fraction)
        
        if dark_field_limit > 0:
            # Multiply the top section of the arena by the dark_gain
            # We apply it to all 3 channels
            self.base_arena[:dark_field_limit, :, :] *= np.sqrt(dark_gain)

        # 4. Red Boundary (5 pixels thick)
        cv2.rectangle(self.base_arena, (0, 0), (self.w, self.h), (0, 0, 1.0), 5)
    def get_crop(self, full_arena, fish_pos):
        # Coordinates in data are often (x, y), but array indexing is [row, col]
        cx, cy = int(fish_pos[0]), int(fish_pos[1])
        
        # Calculate crop bounds while preventing out-of-bounds slicing
        y1 = max(0, min(cy - self.quad, self.h - 2 * self.quad))
        x1 = max(0, min(cx - self.quad, self.w - 2 * self.quad))
        
        return full_arena[y1:y1 + 2 * self.quad, x1:x1 + 2 * self.quad]

def draw_episode_v3(data_file, video_file, num_steps=1000, trim_to_fish=True, showed_region_quad=600):
    with h5py.File(data_file, 'r') as f:
        env_vars = dict(f['env_variables'].attrs)
        fx, fy = np.array(f['fish_x']), np.array(f['fish_y'])
        fa = np.array(f['fish_angle']) # Fish angle in radians
        acts = np.array(f['action'])
        px_all, py_all = np.array(f['prey_x']), np.array(f['prey_y'])
        vis_obs = np.array(f['vis_observation'])
        internal = np.array(f['internal_state'])
        
        pca_data = PCA(n_components=3).fit_transform(detrend(zscore(np.array(f['actor_state'])), axis=0)) if 'actor_state' in f else np.zeros((len(fx), 3))
        renderer = FastRenderer(env_vars, f, True, showed_region_quad)
    if "prey_stim_file" in env_vars:
        with h5py.File(env_vars["prey_stim_file"], 'r') as f:
            prey_stim_locations = f['prey_loc'][:]
    else:
        prey_stim_locations = None
    pca_lims = [(np.percentile(pca_data[:, i], 5), np.percentile(pca_data[:, i], 95)) for i in range(3)]
    
    main_v_size = showed_region_quad * 2
    writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 15, (main_v_size + 400, main_v_size + 250))

    plt.ioff()
    fig = plt.figure(figsize=(4, 4), dpi=100, facecolor='black')
    ax_pca = fig.add_subplot(111, projection='3d', facecolor='black')
    pca_line, = ax_pca.plot([], [], [], color='#00FFFF', lw=1)
    pca_dot = ax_pca.scatter([0], [0], [0], color='white', s=15)
    ax_pca.set_xlim(pca_lims[0]); ax_pca.set_ylim(pca_lims[1]); ax_pca.set_zlim(pca_lims[2])
    ax_pca.axis('off')

    for step in range(min(num_steps, len(fx))):
        canvas = np.zeros((main_v_size + 250, main_v_size + 400, 3), dtype=np.uint8)
        frame = renderer.base_arena.copy()

        # 1. Trajectory with Alpha Fading
        L = min(step, 40)
        if L > 1:
            # Create a dedicated overlay for the tail to handle transparency
            tail_overlay = frame.copy()
            for i in range(step - L, step - 1):
                color = get_bgr_from_action(actions_info[acts[i]]['color'])
                alpha = (i - (step - L)) / L
                cv2.line(tail_overlay, (int(fx[i]), int(fy[i])), (int(fx[i+1]), int(fy[i+1])), color, 3)
            # Blend the tail overlay with the frame based on alpha
            cv2.addWeighted(tail_overlay, 0.7, frame, 0.3, 0, frame)

        # 2. Prey
        for x, y in zip(px_all[step][~np.isnan(px_all[step])], py_all[step][~np.isnan(py_all[step])]):
            cv2.circle(frame, (int(x), int(y)), 6, (0.8, 0.4, 1), -1)
        # Prey stim
        if prey_stim_locations is not None:
            for prey_stims in range(prey_stim_locations.shape[0]):
                x, y = prey_stim_locations[prey_stims, step]
                cv2.circle(frame, (int(x), int(y)), 6, (1, 0.4, 0.8), -1)
        # 3. Fish Shape (Proper Orientation)
        # Using a triangle to show heading clearly
        angle = fa[step]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Fish points: Nose, Left Fin, Right Fin, Tail Tip
        nose = (int(fx[step] + 25 * cos_a), int(fy[step] + 25 * sin_a))
        left = (int(fx[step] - 12 * sin_a), int(fy[step] + 12 * cos_a))
        right = (int(fx[step] + 12 * sin_a), int(fy[step] - 12 * cos_a))
        back = (int(fx[step] - 20 * cos_a), int(fy[step] - 20 * sin_a))
        
        cv2.fillPoly(frame, [np.array([nose, left, back, right])], (0.9, 0.9, 1.0))
        cv2.line(frame, (int(fx[step]), int(fy[step])), nose, (0, 0, 1), 2) # Heading line

        # 4. Action Label
        action_name = actions_info[acts[step]]['name'].split('_')[0]
        cv2.putText(frame, action_name, (int(fx[step])+30, int(fy[step])-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1, 1, 1), 2)

        # 5. UI Assembly
        arena_crop = renderer.get_crop(frame, (fx[step], fy[step]))
        canvas[0:main_v_size, 0:main_v_size] = (np.clip(arena_crop, 0, 1) * 255).astype(np.uint8)

        # Eyes
        for i, offset in enumerate([20, main_v_size // 2 + 20]):
            eye = cv2.resize(vis_obs[step, :, :, i].T, (main_v_size//2 - 40, 160), interpolation=cv2.INTER_NEAREST)

            eye_col = cv2.applyColorMap((eye / 64.0 * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            canvas[main_v_size + 40 : main_v_size + 200, offset : offset + main_v_size//2 - 40] = eye_col

        # PCA Update
        pca_line.set_data(pca_data[max(0,step-60):step, 0], pca_data[max(0,step-60):step, 1])
        pca_line.set_3d_properties(pca_data[max(0,step-60):step, 2])
        pca_dot._offsets3d = (pca_data[step-1:step, 0], pca_data[step-1:step, 1], pca_data[step-1:step, 2])
        fig.canvas.draw()
        pca_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(400, 400, 3)
        canvas[100:500, main_v_size:] = cv2.cvtColor(pca_img, cv2.COLOR_RGB2BGR)

        writer.write(canvas)
    
    writer.release()
    plt.close(fig)

if __name__ == "__main__":
    draw_episode_v3(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 1000)