import pose
import numpy as np
import measurement
import initialization
from numpy.linalg import inv
import constants
from math import cos, sin, tan

class Motion_model(object):
    fc = constants.fc

    def __init__(self):
        self.dt = 1.0/30.0
        print("Motion model is initiated")

    """Updates the angels and rates in the particles by using the motion model '"""
    def rotational_motion_model(self, pose):
        M = matrix_M(pose)
        vnoise = constants.vnoise  # (check eq. 4.15)

        # Get the previous step pose angles and rates
        euler_angles_prev_step = pose.euler_angles # vector of euler_angles [psi theta phi]
        angular_rates_prev_step = pose.angular_rates

        # Calculate the current step angles and rates
        euler_angles = euler_angles_prev_step + np.dot(M, angular_rates_prev_step)*self.dt
        angular_rates = angular_rates_prev_step + vnoise

        # Update particles dictionary
        pose.euler_angles = euler_angles
        pose.angular_rates = angular_rates
        print "Euler angles motion", pose.euler_angles
        print "Angular rates", pose.angular_rates

    """ Estimate translational offset """
    def translational_optimization(self, particle, current_measurements):
        fc = self.fc
        A = np.zeros((2, 1))
        particle_X_map = particle.X_map_dict
        i = 0
        for feature in particle_X_map:
            key = feature.debug_key
            if key in current_measurements.keys():
                x = np.asarray(feature.mean)  # feature x[x y z]
                Rcm = rotation_matrix_CM(particle.pose)
                x_cm = Rcm.dot(x)  # feature in map frame X_c/m [x y z]
                img_coord = np.asarray(current_measurements[key].point)  # image coordinate [u v]
                b11 = (fc*x_cm[0]-img_coord[0]*x_cm[2])
                b21 = (fc*x_cm[0]-img_coord[1]*x_cm[2])
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
            delta_p = inv(A.T.dot(A)).dot(A.T).dot(b)
        except:
            delta_p = np.zeros((3, 1))
        # Update particles
        previous_position = particle.pose.coordinates
        particle.pose.coordinates = delta_p + previous_position

""" Form the rotation matrix that convert camera to map frame(C/M)
Args: particle`s euler_angles array [psi theta phi] """
def rotation_matrix_CM(pose):
    psi = pose.euler_angles[0]
    theta = pose.euler_angles[1]
    phi = pose.euler_angles[2]
    Rcm = np.array([[cos(psi)*cos(theta), sin(psi)*cos(theta), cos(phi)*sin(theta)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*sin(phi)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*cos(phi)]])
    return Rcm

def matrix_M(pose):
    psi = pose.euler_angles[0]
    theta = pose.euler_angles[1]
    phi = pose.euler_angles[2]

    M = np.array([[1, sin(psi)*tan(theta), cos(psi)*tan(theta)],
                  [0, cos(psi), (-1)*sin(psi)],
                  [0, sin(psi)/cos(theta), cos(psi)*cos(theta)]])
    return M






