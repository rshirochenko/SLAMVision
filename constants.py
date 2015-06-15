import numpy as np

# Number of particles
NUMBER_OF_PARTICLES = 3

# Measurements number for the feature X initialization
feature_time_for_init = 4

""" Camera properties """
# Focal length
fc = 12.5  # 12.5 mm

""" Pose properties """
pt0 = [100, 100, 100]

init_coordinates = np.array([1.0, 1.0, 10.0]).reshape((3, 1))
init_euler_angles = np.array([0.1, 0.1, 0.1]).reshape(3, 1)
init_angular_rates = np.array([1.0, 1.0, 1.0]).reshape(3, 1)


""" Feature initialization """
sigma_meas = 0.5
P = np.identity(6)  # TODO: Change P - additive noise covariance


# Vnoise
mean = np.array([0, 0, 0])
cov_w = np.array([[0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1]])
#vnoise = np.random.multivariate_normal(mean, cov_w, (3, 1))
vnoise = np.array([0.01, 0.01, 0.01]).reshape((3, 1))

vnoise_process = 0.1  # fastslam 1.0 noise