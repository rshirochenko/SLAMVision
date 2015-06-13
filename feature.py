import math
import numpy as np
import constants
from numpy.linalg import inv
from math import sin, cos


class Feature(object):
    def __init__(self, mean, covariance, descriptor, debug_coord, debug_key):
        self.mean = mean
        self.covariance = covariance
        self.descriptor = descriptor
        self.debug_coord = debug_coord
        self.debug_key = debug_key


class XMapDictionary(object):
    X_map_dict = []

    def update_X_map_dictionary(self, features_temp, current_measurements):
        features_temp.get_measurements(current_measurements, self.X_map_dict)

    def add_to_X_map_dict(self, feature):
        X_map_dict = self.X_map_dict
        for X_map_feature in X_map_dict:
            if calc_distance_two_descriptors(X_map_feature.descriptor, feature.descriptor) < 100:
                return
            else:
                X_map_dict.append(feature)


class FeaturesTemp(object):
    measurement_dict = {}
    times_observed = {}

    def get_measurements(self, current_measurements, X_map_dict, pose):
        measurement_dict = self.measurement_dict
        cleared_measurements = {}
        # Check the measurement_dict for emptiness
        if not bool(measurement_dict):
            for key in current_measurements:
                if not self.check_Xmap(X_map_dict, current_measurements[key], key):
                    measurement_dict[key] = [current_measurements[key]]
            return

        for key in current_measurements:
            if not self.check_Xmap(X_map_dict, current_measurements[key], key):
                cleared_measurements[key] = current_measurements[key]

        # Add observed measurements to feature temp storage
        for key in cleared_measurements:
            if key in measurement_dict:
                measurement_dict[key].append(cleared_measurements[key])
            else:
                measurement_dict[key] = [cleared_measurements[key]]
        self.check_add_to_Xmap(X_map_dict, cleared_measurements, pose)

    def init_get_measurements(self, current_measurements, X_map_dict, pose):
        measurement_dict = self.measurement_dict
        # Check the measurement_dict for emptiness
        if not bool(measurement_dict):
            for key in current_measurements:
                measurement_dict[key] = [current_measurements[key]]
            self.check_add_to_Xmap(X_map_dict, current_measurements, pose)
            return

    def check_add_to_measurement_dict(self, key, current_measurement):
        measurement_dict = self.measurement_dict
        if key in measurement_dict:
            measurement_dict[key].append(current_measurement)

    def check_Xmap(self, X_map_dict, current_measurement, key):
        result = False
        for feature in X_map_dict:
            if calc_distance_two_descriptors(feature.descriptor, current_measurement.descriptor) < 100:
                feature.debug_key = key
                return True
            else:
                result = False
        return result

    def check_add_to_Xmap(self, X_map_dict, cleared_measurements, pose):
        measurement_dict = self.measurement_dict
        times_observed = self.times_observed
        list_for_deleting = []

        # Counter
        for t in times_observed:
            times_observed[t][0] += 1
        for key in cleared_measurements:
            if key not in times_observed:
                times_observed[key] = [0, 1]
            else:
                times_observed[key][1] += 1

        # Add to X_map
        for key in measurement_dict:
            if times_observed[key][1] == constants.feature_time_for_init:
                mean, covariance = self.EKF_initialization(measurement_dict[key], pose)
                X_map_dict.append(Feature(mean, covariance,
                                              measurement_dict[key][-1].descriptor, measurement_dict[key][-1].point, key))
                list_for_deleting.append(key)
                # Add to delete list if not renewed
            if times_observed[key][0] == times_observed[key][1]:
                list_for_deleting.append(key)

        for d in list_for_deleting:
            del times_observed[d]
            del measurement_dict[d]

    def EKF_initialization(self, measurement_dict_element, pose):
        x = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        covariance = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        i = 1
        At = np.array([])
        bt = np.array([])
        jac_sum = np.zeros((2, 3))
        jac_list = []
        for measurement in measurement_dict_element:
            A, b = self.calc_matrix_Ab_parameters(measurement.point, pose)
            if i == 1:
                At = A
                bt = b
            else:
                At = np.vstack((At, A))
                bt = np.vstack((bt, b))
            x = self.calc_feature(At, bt)
            jacobian = self.calc_jacobian_of_measurement(x)
            jac_list.append(jacobian)
            i += 1
        print "jac_list", jac_list
        covariance = np.zeros((3, 3))
        return x.reshape(3, 1), covariance

    """Args: A, b matrix for the last measurements """
    def calc_feature(self, A, b):
        try:
            x = inv(A.T.dot(A)).dot(A.T).dot(b)
        except np.linalg.linalg.LinAlgError:
            print "Determinant is equal zero"
            x = np.array([0, 0, 0]).reshape(3, 1)
        return x

    """Args: img_coord - 1 measurements SIFT image coordination [u,v] """
    def calc_matrix_Ab_parameters(self, img_coord, pose):
        Rci = rotation_matrix_ci(pose)
        Rmi = rotation_matrix_ci(pose)
        p = pose.coordinates
        fc = constants.fc

        a = Rci.item((0, 0)) * Rmi.item((0, 0)) + Rci.item((0, 1)) * Rmi.item((0, 1)) + Rci.item((0, 2)) * Rmi.item(
            (0, 2))
        b = Rci.item((0, 0)) * Rmi.item((1, 0)) + Rci.item((0, 1)) * Rmi.item((1, 1)) + Rci.item((0, 2)) * Rmi.item(
            (1, 2))
        c = Rci.item((0, 0)) * Rmi.item((2, 0)) + Rci.item((0, 1)) * Rmi.item((2, 1)) + Rci.item((0, 2)) * Rmi.item(
            (2, 2))
        d = Rci.item((0, 0)) * p.item(0) + Rci.item((0, 1)) * p.item(1) + Rci.item((0, 2)) * p.item(2)
        e = Rci.item((1, 0)) * Rmi.item((0, 0)) + Rci.item((1, 1)) * Rmi.item((0, 1)) + Rci.item((1, 2)) * Rmi.item(
            (0, 2))
        f = Rci.item((1, 0)) * Rmi.item((1, 0)) + Rci.item((1, 1)) * Rmi.item((1, 1)) + Rci.item((1, 2)) * Rmi.item(
            (1, 2))
        g = Rci.item((1, 0)) * Rmi.item((2, 0)) + Rci.item((1, 1)) * Rmi.item((2, 1)) + Rci.item((1, 2)) * Rmi.item(
            (2, 2))
        h = Rci.item((1, 0)) * p.item(0) + Rci.item((1, 1)) * p.item(1) + Rci.item((1, 2)) * p.item(2)
        l = Rci.item((2, 0)) * Rmi.item((0, 0)) + Rci.item((2, 1)) * Rmi.item((0, 1)) + Rci.item((2, 2)) * Rmi.item(
            (0, 2))
        m = Rci.item((2, 0)) * Rmi.item((1, 0)) + Rci.item((2, 1)) * Rmi.item((1, 1)) + Rci.item((2, 2)) * Rmi.item(
            (1, 2))
        n = Rci.item((2, 0)) * Rmi.item((2, 0)) + Rci.item((2, 1)) * Rmi.item((2, 1)) + Rci.item((2, 2)) * Rmi.item(
            (2, 2))
        o = Rci.item((2, 0)) * p.item(0) + Rci.item((2, 1)) * p.item(1) + Rci.item((2, 2)) * p.item(2)

        A = np.array([[fc * a - img_coord[0] * l, fc * b - img_coord[0] * m, fc * c - img_coord[0] * n],
                      [fc * e - img_coord[1] * l, fc * f - img_coord[1] * m, fc * g - img_coord[1] * n]])

        b = np.array([[img_coord[0] * o - fc * d],
                      [img_coord[1] * o - fc * h]])
        return A, b

    """Args: List[x[x y z]] - 3D feature coordinates list """
    def calc_mean(self):
        x = 0
        y = 0
        z = 0
        for feature in self.features_list:
            x = x + feature[0]
            y = y + feature[1]
            z = z + feature[2]
        number_of_features = len(self.features_list)
        mean = np.array([x, y, z]) / number_of_features
        return mean

    """Args: x[x y z]] - 3D feature coordinate
    Output: Covariance matrix 3x3 """
    def calc_covariance(self):
        std = self.calc_std_square()
        jac_sum = 0
        for feature in self.features_list:
            jacobian = self.calc_jacobian_of_measurement(feature)
            jac_sum = jac_sum + jacobian.T.dot(jacobian)
        covariance = std * inv(jac_sum)
        return covariance

    def calc_std_square(self):
        std = self.features_list[3].std()
        return np.square(std)

    def calc_jacobian_of_measurement(self, feature):
        feature = np.array(feature).reshape(-1, ).tolist()
        fc = constants.fc
        g11 = fc / feature[2]
        g12 = 0
        g13 = (-1) * fc / (feature[2] * feature[2])
        g21 = 0
        g22 = fc / feature[2]
        g23 = (-1) * float(feature[1]) / (feature[2] * feature[2])
        jacobian = np.array([[g11, g12, g13],
                             [g21, g22, g23]])
        return jacobian

""" Calculate the distance between two 128-demensional sift features descriptors """
def calc_distance_two_descriptors(d1, d2):
    dist = np.linalg.norm(d1 - d2)
    return dist

""" Form the rotation matrix that convert camera to map frame(C/M)
Args: particle`s euler_angles array [psi theta phi] """
def rotation_matrix_mi(pose):
    psi = pose.euler_angles[0]
    theta = pose.euler_angles[1]
    phi = pose.euler_angles[2]

    Rcm = np.array([[cos(psi)*cos(theta), sin(psi)*cos(theta), cos(phi)*sin(theta)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*sin(phi)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*cos(phi)]])
    return Rcm

""" Form the rotation matrix that convert camera to map frame(C/M)
Args: particle`s euler_angles array [psi theta phi] """
def rotation_matrix_ci(pose):
    psi = pose.euler_angles[0]
    theta = pose.euler_angles[1]
    phi = pose.euler_angles[2]

    Rci = np.array([[cos(psi)*cos(theta), sin(psi)*cos(theta), cos(phi)*sin(theta)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*sin(phi)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*cos(phi)]])
    return Rci