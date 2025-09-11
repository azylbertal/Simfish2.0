import matplotlib.pyplot as plt
import h5py
import numpy as np

dir = '/home/asaph/src/simfish2.0/checkpoints_250819/logs/model_evaluation'


# find all files in the directory that start with 'logs_' and end with '.hdf5'
def get_hdf5_files(directory):
    import os
    import re
    files = []
    for file in os.listdir(directory):
        if re.match(r'logs_\d+\.hdf5', file):
            files.append(os.path.join(directory, file))
    return files


files = get_hdf5_files(dir)
start_dist = []
end_dist = []
episode_return = []
actor_steps = []
plt.figure()
for file in files:
    print(file)

    with h5py.File(file, 'r') as f:
        # get the keys of the file
        fish_x = f['fish_x'][:]
        fish_y = f['fish_y'][:]
        salt_location = f['salt_location'][:][0]
        if 'actor_steps' in f:
            episode_return.append(f['episode_return'][()])
            actor_steps.append(f['actor_steps'][()])

        fish_x -= salt_location[0]
        fish_y -= salt_location[1]

        plt.plot(fish_x, fish_y, 'k', alpha=0.15)
        plt.scatter(fish_x[0], fish_y[0], color='green', label='Start', alpha=0.4)
        plt.scatter(fish_x[-1], fish_y[-1], color='red', label='End', alpha=0.4)

        start_dist.append(np.sqrt(fish_x[0]**2 + fish_y[0]**2))
        end_dist.append(np.sqrt(fish_x[-1]**2 + fish_y[-1]**2))

plt.figure()
plt.plot(actor_steps, episode_return, 'o-')
plt.show()

print(f'mean start distance: {np.mean(start_dist)}')
print(f'mean end distance: {np.mean(end_dist)}')

from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(start_dist, end_dist)
print(f't-statistic: {t_stat}, p-value: {p_value}')
