import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib



dir = '/media/gf/DBIO_Bianco_Lab8/temp_data_storage/gcp_output/stage2_sparse_nohdf5/stage2_2/logs/eval_normal'


# find all files in the directory that start with 'logs_' and end with '.hdf5'
def get_hdf5_files(directory):
    import os
    import re
    files = []
    for file in os.listdir(directory):
        if re.match(r'logs_\d+\.hdf5', file):
            files.append(os.path.join(directory, file))
    return files

salt = True
files = get_hdf5_files(dir)
start_dist = []
end_dist = []
episode_return = []
actor_steps = []
all_fish_x = []
all_fish_y = []
init_fish_angle = []
plt.figure()
plt.subplot(1, 2, 1)
for file in files:
    print(file)

    with h5py.File(file, 'r') as f:
        # get the keys of the file
        fish_x = f['fish_x'][:]
        fish_y = f['fish_y'][:]
        fish_angle = f['fish_angle'][:]
        salt_location = f['salt_location'][:][0]
        init_fish_angle.append(fish_angle[0])
        if 'actor_steps' in f:
            episode_return.append(f['episode_return'][()])
            actor_steps.append(f['actor_steps'][()])
        if salt:
            fish_x -= salt_location[0]
            fish_y -= salt_location[1]
        all_fish_x.append(fish_x)
        all_fish_y.append(fish_y)
        plt.plot(fish_x, fish_y, 'k', alpha=0.2, linewidth=0.3)
        plt.scatter(fish_x[0], fish_y[0], color='green', label='Start', alpha=0.3, s=5)
        # plt.scatter(fish_x[-1], fish_y[-1], color='red', label='End', alpha=0.4, s=5)

# invert y axis so that up is positive
# plt.xlabel('Fish X Position')
# plt.ylabel('Fish Y Position')
plt.title('Fish Trajectories')
if salt:
    plt.scatter(0, 0, color='orange', label='Salt Location', alpha=0.5, s=150)
else:
        plt.fill_between([0, 2500], 0, 2500 * 0.3, color='k', alpha=0.5)
        # invert the y axis
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)

        plt.plot([0, 2500, 2500, 0, 0], [0, 0, 2500, 2500, 0], color='k', label='Arena Edge')
# plt.legend()
# plt.axis('equal')
plt.gca().invert_yaxis()
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().set_aspect('equal', adjustable='box')

# concatenate all fish positions
all_fish_x = np.concatenate(all_fish_x)
all_fish_y = np.concatenate(all_fish_y)
# remove axes and box

# create a kernel density estimate plot of all fish positions
if False:
    import seaborn as sns
    plt.subplot(1, 2, 2)
    sns.kdeplot(x=all_fish_x, y=all_fish_y, fill=True, cmap='Blues', bw_adjust=0.5)
    # plt.xlabel('Fish X Position')
    # plt.ylabel('Fish Y Position')
    plt.title('KDE of Fish Positions')
    if salt:
        plt.scatter(0, 0, color='orange', label='Salt Location', alpha=0.5, s=150)
    else:
        plt.plot([0, 2500, 2500, 0, 0], [0, 0, 2500, 2500, 0], color='red', label='Arena Edge', alpha=0.5)
        plt.plot([0, 2500], [750, 750], color='blue', alpha=0.5)
    # plt.axis('equal')

    plt.gca().invert_yaxis()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

plt.show()