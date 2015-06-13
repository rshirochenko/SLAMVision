import numpy as np

# Number of particles
NUMBER_OF_PARTICLES = 3

# Measurements number for the feature X initialization
feature_time_for_init = 4

""" Camera properties """
# Focal strength
fc = 0.1

""" Pose properties """
pt0 = [100, 100, 100]

init_coordinates = np.array([1.0, 1.0, 10.0]).reshape((3, 1))
init_euler_angles = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
init_angular_rates = np.array([1.0, 1.0, 1.0]).reshape(3, 1)


""" Feature initialization """
sigma_meas = 0.5