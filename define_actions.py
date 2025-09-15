import h5py
import numpy as np
from scipy.stats import zscore, multivariate_normal
from scipy import odr

PIXEL_SIZE = 0.058  # mm per pixel, as per the original code
ALL_BOUT_NAMES = ['SCS', 'LCS', 'BS', 'O-bend', 'J-turn', 'SLC', 'Slow1', 'RT', 'Slow2', 'LLC', 'AS', 'SAT', 'HAT']
IS_TURN = [False, False, False, True, True, True, False, True, False, True, False, True, True]
IS_CAPTURE = [True, True, False, False, False, False, False, False, False, False, False, False, False]
# get colors from tableau 20
COLORS = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255), (44/255, 160/255, 44/255), (214/255, 39/255, 40/255),
          (148/255, 103/255, 189/255), (140/255, 86/255, 75/255), (227/255, 119/255, 194/255), (127/255, 127/255, 127/255),
          (188/255, 189/255, 34/255), (23/255, 190/255, 207/255), (174/255, 199/255, 232/255), (255/255, 187/255, 120/255),
          (152/255, 223/255, 138/255)]
BOUT_ENERGY = np.array([0.03, np.nan, np.nan, np.nan, 0.04, np.nan, 0.01, 0.15, 0.15, np.nan, 0.025, np.nan, 0.15])

     
def get_angles_and_distances(times, head_pos, orientation):
    distance = np.zeros(times.shape[0])
    ori_change = np.zeros(times.shape[0])
    for i in range(times.shape[0]):
        this_duration = int(times[i, 1] - times[i, 0]) + 9
        if this_duration >= 175:
            distance[i] = np.nan
            ori_change[i] = np.nan
        else:
            head_pos_change = head_pos[i, this_duration, :] - head_pos[i, 0, :]
            distance[i] = PIXEL_SIZE * (head_pos_change[0] ** 2 + head_pos_change[1] ** 2) ** 0.5
            ori_change[i] = (orientation[i, this_duration] - orientation[i, 0])
            ori_change[i] = np.arctan2(np.sin(ori_change[i]), np.cos(ori_change[i]))  # Normalize to [-pi, pi]

            if distance[i] < 0.01:
                distance[i] = np.nan
                ori_change[i] = np.nan
    return distance, ori_change

def extract_bout_sample(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        # Extract the bout distribution data
        lengths = f['MetaData']['lengths_data'][:].astype(int)
        bout_types = f['bout_types'][:]
        head_pos = f['head_pos'][:]
        orientation = f['orientation_smooth'][:]
        times_bouts = f['times_bouts'][:]
    
    n_fish = len(lengths)
    bout_distances = [np.array([])] * 13
    bout_angles = [np.array([])] * 13
    for fish in range(n_fish):
        this_length = lengths[fish]
        distance, ori_change = get_angles_and_distances(times_bouts[fish][:this_length], head_pos[fish][:this_length], orientation[fish][:this_length])
        this_bout_types = bout_types[fish][:this_length]
        for bt in range(1, 14):
            this_distances = distance[this_bout_types == bt]
            this_ori_change = ori_change[this_bout_types == bt]
            bout_distances[bt - 1] = np.append(bout_distances[bt - 1], this_distances)
            bout_angles[bt - 1] = np.append(bout_angles[bt - 1], this_ori_change)

    return bout_distances, bout_angles

class Actions:

    def __init__(self, h5_file_path=None, bouts_to_save=None):
        if h5_file_path is None:
            self.actions = []
        else:
            self.actions = self.get_extracted_actions(h5_file_path, bouts_to_save=bouts_to_save)

    def get_action(self, action_name):
        for action in self.actions:
            if action['name'] == action_name:
                return action
        raise ValueError(f"Action {action_name} not found in actions.")
    
    def display_actions(self):

        xx, yy = np.mgrid[0:20:.1, -3:3:.01]
        pos = np.dstack((xx, yy))
        plt.figure()
        for id, action in enumerate(self.actions):
            rv = multivariate_normal(action['mean'], action['cov'], allow_singular=True)
            pdf = rv.pdf(pos)
            half_max = np.max(pdf) / 2
            CS = plt.contour(xx, yy, pdf, levels=[half_max], alpha=0.5, colors=[action['color']])
            label = f'{id}-{action["name"]}'
            plt.clabel(CS, CS.levels, fmt={CS.levels[0]:label}, fontsize=8, inline_spacing=1)
        plt.xlabel('Distance (mm)')
        plt.ylabel('Angle (radians)')

    def get_all_actions(self):
        return self.actions

    def get_extracted_actions(self, h5_file_path, bouts_to_save=None):

        """
        Extracts the actions from the h5 file and returns a list of dictionaries containing the action name, mean, covariance, whether it is a turn, and whether it is a capture.
        """

        if bouts_to_save is None:
            bouts_to_save = ALL_BOUT_NAMES
        print(f"Extracting actions from {h5_file_path} with bouts to save: {bouts_to_save}")
        dists, angles = extract_bout_sample(h5_file_path)
        all_means = []
        all_actions = []
        for i in range(13):
            eligble_indices = (~np.isnan(dists[i])) & (~np.isnan(angles[i]))
            eligble_angles = angles[i][eligble_indices]
            eligble_dists = dists[i][eligble_indices]

            if IS_TURN[i]:
                eligble_angles = np.abs(eligble_angles)
            # remove outliers with outlier_thresh standard deviations
            outlier_thresh = 5
            angles_zscore = zscore(eligble_angles)
            dists_zscore = zscore(eligble_dists)
            within_threshold_indices = (np.abs(angles_zscore) < outlier_thresh) & (np.abs(dists_zscore) < outlier_thresh)
            eligble_angles = eligble_angles[within_threshold_indices]
            eligble_dists = eligble_dists[within_threshold_indices]
            
            combined = np.array([eligble_dists, eligble_angles])
            mean = np.mean(combined, axis=1)
            cov = np.cov(combined)  
            all_means.append(mean)
            if not IS_TURN[i]:
                mean[1] = 0.0 # Set mean angle to 0 for non-turn bouts
                cov[0, 1] = 0.0
                cov[1, 0] = 0.0
                if ALL_BOUT_NAMES[i] in bouts_to_save:
                    all_actions.append({'name': ALL_BOUT_NAMES[i], 'mean': mean, 'cov': cov, 'is_turn': IS_TURN[i],\
                                         'is_capture': IS_CAPTURE[i], 'color': COLORS[i]})
            else:
                if ALL_BOUT_NAMES[i] in bouts_to_save:
                    opposite_mean = mean.copy()
                    opposite_mean[1] = -mean[1]
                    opposite_cov = cov.copy()
                    opposite_cov[0, 1] *= -1
                    opposite_cov[1, 0] *= -1  # Keep the covariance the same but flipped

                    all_actions.append({'name': ALL_BOUT_NAMES[i]+"_R", 'mean': mean, 'cov': cov, 'is_turn': IS_TURN[i],\
                                         'is_capture': IS_CAPTURE[i], 'color': COLORS[i]})
                    all_actions.append({'name': ALL_BOUT_NAMES[i]+"_L", 'mean': opposite_mean, 'cov': opposite_cov, 'is_turn': IS_TURN[i],\
                                         'is_capture': IS_CAPTURE[i], 'color': COLORS[i]})
        # fit a linear regression between the means and the energy
        all_means = np.array(all_means)
        non_nan_energy = BOUT_ENERGY[~np.isnan(BOUT_ENERGY)]
        non_nan_means = all_means[~np.isnan(BOUT_ENERGY)]
        data = odr.Data(non_nan_means.T, non_nan_energy)
        odr_obj = odr.ODR(data, odr.multilinear)
        output = odr_obj.run()
        reg = output.beta
        
        non_nan_preds = np.dot(non_nan_means, reg[1:]) + reg[0]

        plt.figure()
        plt.scatter(non_nan_energy, non_nan_preds, color='black', label='Predicted Energy')
        plt.xlabel('Goodhill Energy')
        plt.ylabel('Predicted Energy from Linear Regression')
        plt.plot([0, 0.2], [0, 0.2], color='red', linestyle='--', label='y=x')
        plt.title('Predicted vs Actual Goodhill Energy')
        plt.axis('equal')
        print("Linear Regression Coefficients:")
        print(f"Intercept: {reg[0]}")
        print(f"Energy coefficients: {reg[1:]}")
         

        return all_actions

    def sharpen_distributions(self, narrowing_coefficient=3, capture_narrowing_coefficient=10):
        """
        Sharpens the distributions by dividing the covariance matrix by a narrowing coefficient.
        This is useful for making the actions more distinct.
        """
        for action in self.actions:
            if action['is_capture']:
                action['cov'] /= capture_narrowing_coefficient
                action['cov'][1, 1] /= capture_narrowing_coefficient  # reduce angle variance for capture actions
            else:
                action['cov'] /= narrowing_coefficient

    def to_hdf5(self, file_path):
        """
        Saves the actions to an HDF5 file.
        """
        with h5py.File(file_path, 'w') as f:
            for i, action in enumerate(self.actions):
                group = f.create_group(action['name'])
                group.create_dataset('mean', data=action['mean'])
                group.create_dataset('cov', data=action['cov'])
                group.attrs['is_turn'] = action['is_turn']
                group.attrs['is_capture'] = action['is_capture']
                group.attrs['id'] = i
                group.attrs['color'] = action['color']

    def from_hdf5(self, file_path):
        """
        Loads the actions from an HDF5 file.
        """
        self.actions = []
        ids = []
        with h5py.File(file_path, 'r') as f:
            for group_name in f.keys():
                group = f[group_name]
                action = {
                    'name': group_name,
                    'mean': group['mean'][:],
                    'cov': group['cov'][:],
                    'is_turn': group.attrs['is_turn'],
                    'is_capture': group.attrs['is_capture'],
                    'color': group.attrs['color']
                }
                ids.append(group.attrs['id'])
                self.actions.append(action)
        # sort actions by id
        self.actions = [x for _, x in sorted(zip(ids, self.actions), key=lambda pair: pair[0])]
        print(f"Loaded {len(self.actions)} actions from {file_path}")

    def get_opposing_dict(self):
        """
        Returns a dictionary mapping each action to its opposing action.
        """
        opposing_dict = {}
        for id, action in enumerate(self.actions):
            if action['is_turn']:
                opposing_name = action['name'].replace('_R', '_L') if '_R' in action['name'] else action['name'].replace('_L', '_R')
                opposing_id = [i for i, a in enumerate(self.actions) if a['name'] == opposing_name][0]
                opposing_dict[id] = opposing_id
        return opposing_dict
    def add_null_action(self):
        """
        Adds a null action with zero mean and covariance (no movement).
        """
        null_action = {
            'name': 'Null',
            'mean': np.array([0.0, 0.0]),
            'cov': np.array([[0.0, 0.0], [0.0, 0.0]]),
            'is_turn': False,
            'is_capture': False,
            'color': (0.5, 0.5, 0.5)
        }
        self.actions.append(null_action)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    h5_file_path = "filtered_jmpool_kin.h5"  # Bout data file path

    actions = Actions(h5_file_path, bouts_to_save=None)  # Use None to extract all bouts
    actions.sharpen_distributions(narrowing_coefficient=3, capture_narrowing_coefficient=10)
    actions.display_actions()
    actions.add_null_action()
    print(f'opposing_dict: {actions.get_opposing_dict()}')
    actions.to_hdf5("actions_all_bouts_with_null.h5")
    actions.from_hdf5("actions_all_bouts_with_null.h5")
    actions.display_actions()


    plt.show()
