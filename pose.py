import math
import numpy as np
import initialization


class Particle(object):
    def __init__(self, weight, pose, X_map):
        self.weight = weight
        self.pose = pose
        self.X_map = X_map


class Pose(object):
    def __init__(self, euler_angles, angular_rates, position):
        self.euler_angles = euler_angles  # euler_angles array [psi theta phi]
        self.angular_rates = angular_rates  # angular_rate [wx wy wz]
        self.position = position  # position array [px py pz]


"""Make dump particles dictionary"""
def make_particles_dict(X_map):
    particles = dict()
    n = len(X_map)
    for i in xrange(n):
        weight = 1/n
        particles[i] = Particle(weight, Pose(np.random.rand(3), np.random.rand(3), np.random.rand(3)), X_map)
    return particles


def main():
    particles_dict = make_particles_dict(2)
    print particles_dict

if __name__ =='__main__':main()

