import numpy as np
import random

"""Video properties """
fps = 30  # fps of the video
dt = 1.0/fps

""" Algorithm properties """
# Number of particles
NUMBER_OF_PARTICLES = 3

# Measurements number for the feature X initialization
feature_time_for_init = 4

""" Camera properties """
# Focal length
fc = 12.5  # 12.5 mm

""" Pose properties """
pt0 = [100, 100, 100]

init_coordinates = np.array([1.0, 1.0, 1.0]).reshape((3, 1))
init_euler_angles = np.array([0, 0, 0]).reshape(3, 1)
init_angular_rates = np.array([0, 0, 0]).reshape(3, 1)


""" Feature initialization """
sigma_meas = 0.5
P = np.identity(9)  # TODO: Change P - additive noise covariance


# Vnoise
sigma_vn = 0.01
mean = np.array([random.gauss(0, sigma_vn), random.gauss(0, sigma_vn), random.gauss(0, sigma_vn)])
cov_w = np.array([[sigma_vn**2, 0, 0],
                  [0, sigma_vn**2, 0],
                  [0, 0, sigma_vn**2]])

""" Motion model """
sigma_vnoise = random.gauss(0, sigma_vn)

