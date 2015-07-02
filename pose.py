import numpy as np

class Particle(object):
    """
    Class describes particle in particle filter
    Attributes:
        weight: weight of the particle
        pose: proposed pose of the particle
        X_map: features observed in the current frame
        X_map_dict: map(X) with all features
    """
    def __init__(self, weight, pose, X_map, X_map_dict):
        self.weight = weight
        self.pose = pose
        self.X_map = X_map
        self.X_map_dict = X_map_dict


class Pose(object):
    """
    Class describes pose (st)
    Attributes:
        euler_angles: euler_angles array [psi theta phi]
        angular_rates: angular_rate [wx wy wz]
        coordinates: position array [px py pz]
    """
    def __init__(self, euler_angles, angular_rates, coordinates):
        self.euler_angles = euler_angles  # euler_angles array [psi theta phi]
        self.angular_rates = angular_rates  # angular_rate [wx wy wz]
        self.coordinates = coordinates  # position array [px py pz]
