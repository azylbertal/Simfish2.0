import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#from Analysis.Calibration.ActionSpace.draw_angle_dist_old import convert_action_to_bout_id

"""To create the final form of all the bouts, and display them for ease - 2D Gaussian distributions in all cases."""
def convert_action_to_bout_id(action):
    if action == 0:
        return 8
    elif action == 1 or action == 2:
        return 7
    elif action == 3:
        return 0
    elif action == 4 or action == 5:
        return 4
    elif action == 7 or action == 8:
        return 5
    elif action == 9:
        return 10
    elif action == 10 or action == 11:
        return 44

def get_action_name_unlateralised(action_num):
    if action_num == 0:
        action_name = "Slow2"
    elif action_num == 1:
        action_name = "RT"
    elif action_num == 2:
        action_name = "RT"
    elif action_num == 3:
        action_name = "sCS"
    elif action_num == 4:
        action_name = "J-turn"
    elif action_num == 5:
        action_name = "J-turn"
    elif action_num == 6:
        action_name = "Rest"
    elif action_num == 7:
        action_name = "SLC"
    elif action_num == 8:
        action_name = "SLC"
    elif action_num == 9:
        action_name = "AS"
    else:
        action_name = "None"
    return action_name

def get_bout_data(action_num):
    bout_id = convert_action_to_bout_id(action_num)
    bout_id += 1

    mat = scipy.io.loadmat("/home/asaph/src/Simfish2.0/analysis/bouts.mat")

    bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
    bout_inferred_final_array = mat["BoutInfFinalArray"]

    angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
    distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
    distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

    distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

    bouts = bout_inferred_final_array[:, 133].astype(int)

    relevant_bouts = (bouts == bout_id)
    angles = np.absolute(angles[relevant_bouts])
    distance = distance[relevant_bouts]

    angles *= (np.pi/180)

    return distance, angles


def create_new_bout_slow2(narrowing_coefficient):
    """Separates into two, narrowed distributions that preserves covariance between distance and angle."""
    distance_flattening_threshold = 15
    angle_flattening_threshold = 0.25

    c_start_distance, c_start_angle = get_bout_data(0)

    all_bouts = np.concatenate((np.expand_dims(c_start_distance, 1), np.expand_dims(c_start_angle, 1)), axis=1)
    valid_bouts = (all_bouts[:, 0] < distance_flattening_threshold)
    all_bouts = all_bouts[valid_bouts]

    valid_bouts = (all_bouts[:, 1] < angle_flattening_threshold)
    all_bouts = all_bouts[valid_bouts]

    # Split distributions first (minimal bias)
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(all_bouts)

    plt.scatter(c_start_distance, c_start_angle)
    plt.scatter(all_bouts[:, 0], all_bouts[:, 1])#, c=kmeans.labels_)
    plt.savefig(f"/home/asaph/src/Simfish2.0/analysis/bouts/Slow2-Discarded.jpg")
    plt.clf()
    plt.close()

    # As is a forward bout, mirror the sign of the angles for better estimation of mean.
    negative_angle_bouts = np.concatenate((all_bouts[:, 0:1], all_bouts[:, 1:2]*-1), axis=1)
    all_bouts = np.concatenate((all_bouts, negative_angle_bouts), axis=0)

    generate_multivariates_from_bouts(all_bouts, narrowing_coefficient, "Slow2")


def create_new_bout_c_start(narrowing_coefficient):
    """Separates into two, narrowed distributions that preserves covariance between distance and angle."""
    distance_flattening_threshold = 15
    angle_flattening_threshold = 2

    c_start_distance, c_start_angle = get_bout_data(7)

    all_bouts = np.concatenate((np.expand_dims(c_start_distance, 1), np.expand_dims(c_start_angle, 1)), axis=1)
    valid_bouts = (all_bouts[:, 0] < distance_flattening_threshold)
    all_bouts = all_bouts[valid_bouts]

    valid_bouts = (all_bouts[:, 1] < angle_flattening_threshold)
    all_bouts = all_bouts[valid_bouts]

    # Split distributions first (minimal bias)
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(all_bouts)

    plt.scatter(c_start_distance, c_start_angle)
    plt.scatter(all_bouts[:, 0], all_bouts[:, 1])#, c=kmeans.labels_)
    plt.savefig(f"/home/asaph/src/Simfish2.0/analysis/bouts/C-Start-Discarded.jpg")
    plt.clf()
    plt.close()

    generate_multivariates_from_bouts(all_bouts, narrowing_coefficient, "C-Start")


def create_new_bouts_j_turns(narrowing_coefficient):
    """Separates into two, narrowed distributions that preserves covariance between distance and angle."""

    j_turn_distance, j_turn_angle = get_bout_data(4)

    all_bouts = np.concatenate((np.expand_dims(j_turn_distance, 1), np.expand_dims(j_turn_angle, 1)), axis=1)

    # Split distributions first (minimal bias)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(all_bouts)

    angle_means = [np.mean(all_bouts[kmeans.labels_ == i, 1]) for i in range(3)]
    low_cluster = np.argmin(angle_means)

    j_turn_1 = all_bouts[kmeans.labels_ == low_cluster]
    j_turn_2 = all_bouts[kmeans.labels_ != low_cluster]

    plt.scatter(j_turn_1[:, 0], j_turn_1[:, 1])
    plt.scatter(j_turn_2[:, 0], j_turn_2[:, 1])
    plt.savefig(f"/home/asaph/src/Simfish2.0/analysis/bouts/J-Turn-Scatter-Split.jpg")
    plt.clf()
    plt.close()

    generate_multivariates_from_bouts(j_turn_1, narrowing_coefficient, "J-Turn-1")
    generate_multivariates_from_bouts(j_turn_2, narrowing_coefficient, "J-Turn-2")


def generate_multivariates_from_bouts(bouts, narrowing_coefficient, bout_name):
    mean = np.mean(bouts, axis=0)
    cov = np.cov(bouts, rowvar=0)

    # Narrow distributions as decided.
    cov /= narrowing_coefficient

    print(f"""{bout_name}
    Mean: {mean}
    Cov: {cov}
    """)

    #            VISUALISATIONS

    # Display real vs Geneated Data
    new_data = np.random.multivariate_normal(mean, cov, bouts.shape[0])

    plt.scatter(bouts[:, 0], bouts[:, 1])
    plt.scatter(new_data[:, 0], new_data[:, 1])
    plt.xlabel("Distance (mm)")
    plt.ylabel("Angle (radians)")
    plt.savefig(f"/home/asaph/src/Simfish2.0/analysis/bouts/{bout_name}-Real-Vs-Generated-Data.jpg")
    plt.clf()
    plt.close()

    # As a histogram
    new_data = np.random.multivariate_normal(mean, cov, 100000)

    plt.hist2d(new_data[:, 0], new_data[:, 1], bins=100)
    plt.xlabel("Distance (mm)")
    plt.ylabel("Angle (radians)")
    plt.savefig(f"/home/asaph/src/Simfish2.0/analysis/bouts/{bout_name}-Dense-Generated-Data.jpg")
    plt.clf()
    plt.close()


def create_new_dist_gaussian(action_num, narrowing_coefficient, distance_flattening_threshold=None,
                             angle_flattening_threshold=None, forward_swim=False):

    bout_name = get_action_name_unlateralised(action_num)

    distance, angle = get_bout_data(action_num)
    bouts = np.concatenate((np.expand_dims(distance, 1), np.expand_dims(angle, 1)), axis=1)

    # Process to remove undesired outliers
    if distance_flattening_threshold is not None:
        valid_bouts = (bouts[:, 0] < distance_flattening_threshold)
        bouts = bouts[valid_bouts]
    if angle_flattening_threshold is not None:
        valid_bouts = (bouts[:, 1] < angle_flattening_threshold)
        bouts = bouts[valid_bouts]

    if forward_swim:
        # As is a forward bout, mirror the sign of the angles for better estimation of mean.
        negative_angle_bouts = np.concatenate((bouts[:, 0:1], bouts[:, 1:2] * -1), axis=1)
        bouts = np.concatenate((bouts, negative_angle_bouts), axis=0)

    generate_multivariates_from_bouts(bouts, narrowing_coefficient, bout_name)

    # # Computation of mean and covariance parameters for new distributions
    # mean = np.mean(bouts, axis=0)
    # cov = np.cov(bouts, rowvar=0)
    #
    # # Narrow distributions as decided.
    # cov /= narrowing_coefficient
    #
    # #            VISUALISATIONS
    #
    # # Display real vs Geneated Data
    # new_data = np.random.multivariate_normal(mean, cov, bouts.shape[0])
    #
    # plt.scatter(bouts[:, 0], bouts[:, 1])
    # plt.scatter(new_data[:, 0], new_data[:, 1])
    # plt.xlabel("Distance (mm)")
    # plt.ylabel("Angle (radians)")
    # plt.savefig(f"Spatial-Density-Fish-Prey-Position-Metrics/Final-Bouts/{bout_name}-Real-Vs-Generated-Data.jpg")
    # plt.clf()
    # plt.close()
    #
    # # As a histogram
    # new_data = np.random.multivariate_normal(mean, cov, 100000)
    #
    # plt.hist2d(new_data[:, 0], new_data[:, 1], bins=100)
    # plt.xlabel("Distance (mm)")
    # plt.ylabel("Angle (radians)")
    # plt.savefig(f"Spatial-Density-Fish-Prey-Position-Metrics/Final-Bouts/{bout_name}-Dense-Generated-Data.jpg")
    # plt.clf()
    # plt.close()

    # Generating a meshgrid complacent with the 3-sigma boundary
    # distr = multivariate_normal(cov=cov, mean=mean)
    # mean_1, mean_2 = mean[0], mean[1]
    # sigma_1, sigma_2 = cov[0, 0], cov[1, 1]
    #
    # x = np.linspace(-6 * sigma_1, 6 * sigma_1, num=100) + mean_1
    # y = np.linspace(-10 * sigma_2, 10 * sigma_2, num=100) + mean_2
    # X, Y = np.meshgrid(x, y)
    #
    # # Generating the density function for each point in the meshgrid
    # pdf = np.zeros(X.shape)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
    # plt.contourf(X, Y, pdf, cmap='viridis')
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    # sCS
    create_new_dist_gaussian(3, 10, angle_flattening_threshold=2, forward_swim=True)

    # AS
    create_new_dist_gaussian(9, 3, forward_swim=True)

    # RT
    create_new_dist_gaussian(1, 3, distance_flattening_threshold=10)

    # J-turn
    create_new_bouts_j_turns(3)

    # C-starts
    create_new_bout_c_start(3)

    # Slow2
    create_new_bout_slow2(3)
