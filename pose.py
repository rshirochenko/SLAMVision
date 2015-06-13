import math
import numpy as np
import initialization


class Particle(object):
    def __init__(self, weight, pose, X_map, X_map_dict):
        self.weight = weight
        self.pose = pose
        self.X_map = X_map
        self.X_map_dict = X_map_dict


class Pose(object):
    def __init__(self, euler_angles, angular_rates, coordinates):
        self.euler_angles = euler_angles  # euler_angles array [psi theta phi]
        self.angular_rates = angular_rates  # angular_rate [wx wy wz]
        self.coordinates = coordinates  # position array [px py pz]


"""Make dump particles dictionary"""
def make_particles_dict(X_map):
    particles = dict()
    n = len(X_map)
    for i in xrange(n):
        weight = 1/n
        new_pose = Pose(np.random.rand(3), np.random.rand(3), np.random.rand(3))
        particles[i] = Particle(weight, new_pose, X_map)
    return particles




