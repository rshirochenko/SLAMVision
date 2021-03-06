import numpy as np
from numpy.linalg import inv
import constants
from math import cos, sin, tan
import random

class Motion_model(object):

    fc = constants.fc  # focal length
    dt = constants.dt  # time between frames (1/fps)

    def rotational_motion_model(self, pose):
        """ Update the angels and rates in the particles by using the motion model. """
        M = matrix_M(pose)

        sigma_vnoise = constants.sigma_vnoise  # the rotational noise term (zero-mean random angular acceleration)
        vnoise = np.array([random.gauss(0, sigma_vnoise),
                   random.gauss(0, sigma_vnoise),
                   random.gauss(0, sigma_vnoise)]).reshape((3, 1))

        # Get the previous step pose angles and rates
        euler_angles_prev_step = pose.euler_angles # vector of euler_angles [psi theta phi]
        angular_rates_prev_step = pose.angular_rates

        # Calculate the current step angles and rates
        euler_angles = euler_angles_prev_step + np.dot(M, angular_rates_prev_step)*self.dt
        angular_rates = angular_rates_prev_step + vnoise

        # Update particles dictionary
        pose.euler_angles = euler_angles
        pose.angular_rates = angular_rates


    def translational_optimization(self, particle, current_measurements):
        """ Estimate translational offset. """
        fc = self.fc
        A = np.zeros((2, 1))
        particle_X_map = particle.X_map_dict
        i = 0
        for feature in particle_X_map:
            key = feature.feature_key
            if key in current_measurements.keys():
                x = np.asarray(feature.mean)  # feature x[x y z]
                Rcm = rotation_matrix_CM(particle.pose)
                x_cm = Rcm.dot(x) + particle.pose.coordinates  # feature in map frame X_c/m [x y z]
                img_coord = np.asarray(current_measurements[key].point)  # image coordinate [u v]
                b11 = (fc*x_cm[0]-img_coord[0]*x_cm[2])
                b21 = (fc*x_cm[1]-img_coord[1]*x_cm[2])
                bx = np.array([b11, b21])
                if b11 == 0:
                    bx = np.zeros((2, 1))
                Ax = np.array([[(-1)*fc, 0, img_coord[0]],
                               [0, (-1)*fc, img_coord[1]]])
                if i == 0:
                    A = Ax
                    b = bx
                else:
                    A = np.vstack((A, Ax))
                    b = np.vstack((b, bx))
                i += 1
        try:
            delta_p = inv(A.T.dot(A)).dot(A.T).dot(b) # translation offset
        except:
            delta_p = np.zeros((3, 1))
        # Update particles
        previous_position = particle.pose.coordinates
        particle.pose.coordinates = delta_p + previous_position

def rotation_matrix_CM(pose):
    """
    Form the rotation matrix that converts features from camera to map frame(C/M)
    Args:
        pose: particle`s euler_angles array [psi theta phi]
    """
    psi = pose.euler_angles[0]
    theta = pose.euler_angles[1]
    phi = pose.euler_angles[2]
    Rcm = np.array([[cos(psi)*cos(theta), sin(psi)*cos(theta), cos(phi)*sin(theta)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*sin(phi)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*cos(phi)]])
    return Rcm

def matrix_M(pose):
    """
    Matrix M that maps angular rates to Euler angle derivatives(Eq. 4.14)
    Args:
        pose: particle`s euler_angles array [psi theta phi]
    """
    psi = pose.euler_angles[0]
    theta = pose.euler_angles[1]
    phi = pose.euler_angles[2]

    M = np.array([[1, sin(psi)*tan(theta), cos(psi)*tan(theta)],
                  [0, cos(psi), (-1)*sin(psi)],
                  [0, sin(psi)/cos(theta), cos(psi)*cos(theta)]])
    return M
