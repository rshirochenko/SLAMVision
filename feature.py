import math
import numpy as np
import constants
from numpy.linalg import inv
from math import sin, cos


class Feature(object):
    """
    Class describes the one feature(X) object
    Attributes:
        mean: mean vector [mean_x mean_y mean_z] for the feature
        covariance: covariance matrix 3x3
        descriptor: SIFT 128-vector descriptor for the correspondence between measurement(Z) and feature(X)
        debug_coord: coordinate of the corresponded measurement(Z)
        feature_key: index in the measurements table K
    """
    def __init__(self, mean, covariance, descriptor, debug_coord, feature_key):
        self.mean = mean
        self.covariance = covariance
        self.descriptor = descriptor
        self.debug_coord = debug_coord
        self.feature_key = feature_key


class XMapDictionary(object):
    """
    Class describes X map
    Attributes:

    """
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


class FeaturesCache(object):
    measurement_dict = {}
    times_observed = {}

    def get_measurements(self, current_measurements, X_map_dict, pose):
        """
        Calculate the measurements for the current frame
        Args:
            current_measurements: current SIFTs measurements in the frame
            X_map_dict: features map (X)
            pose: pose for the current frame
        """
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
        """
        Calculate the measurements for the current frame
        Args:
            current_measurements: current SIFTs measurements in the frame
            X_map_dict: features map (X)
            pose: pose for the current frame
        """
        measurement_dict = self.measurement_dict
        # Check the measurement_dict for emptiness
        if not bool(measurement_dict):
            for key in current_measurements:
                measurement_dict[key] = [current_measurements[key]]
            self.check_add_to_Xmap(X_map_dict, current_measurements, pose)
            return

    def check_add_to_measurement_dict(self, key, current_measurement):
        """
        Args:
            key: index of dictionary in table K of SIFTs measurements
            pose: pose for the current frame
        """
        measurement_dict = self.measurement_dict
        if key in measurement_dict:
            measurement_dict[key].append(current_measurement)

    def check_Xmap(self, X_map_dict, current_measurement, key):
        """
        Check if the converted measurement to feature will be in the X map
        Args:
            X_map_dict: X map of features (dictionary)
            current_measurement: SIFTs measurements from the current frame(image)
            key: index in the table K (dictionary data structure)
        Returns:
            result: boolean variable, if measurement will be found, then return True
        """
        result = False
        for feature in X_map_dict:
            if calc_distance_two_descriptors(feature.descriptor, current_measurement.descriptor) < 100:
                feature.feature_key = key
                return True
            else:
                result = False
        return result

    def check_add_to_Xmap(self, X_map_dict, cleared_measurements, pose):
        """
        Convert measurement to feature and add feature to X map
        Args:
            X_map_dict: X map of features (dictionary)
            cleared_measurement: measurements that were not in the X map before
            pose: pose for the current frame
        """
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
            try:
                times_observed[key]
            except KeyError:
                list_for_deleting.append(key)
                continue
            if times_observed[key][1] == constants.feature_time_for_init:
                mean, covariance = self.EKF_initialization(measurement_dict[key], pose)
                X_map_dict.append(Feature(mean, covariance,
                                              measurement_dict[key][-1].descriptor, measurement_dict[key][-1].point, key))
                list_for_deleting.append(key)
                # Add to delete list if not renewed
            if times_observed[key][0] == times_observed[key][1]:
                list_for_deleting.append(key)

        for d in list_for_deleting:
            try:
                del times_observed[d]
                del measurement_dict[d]
            except KeyError:
                pass

    def EKF_initialization(self, measurement_dict_element, pose):
        """
        Calculate the EKF for the new feature
        Args:

        """
        x = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        covariance = np.zeros((3, 3))
        i = 1
        At = np.array([])
        bt = np.array([])
        jac_list = []
        for measurement in measurement_dict_element:
            A, b = self.calc_matrix_Ab_parameters(measurement.point, pose)
            if i == 1:
                At = A
                bt = b
            else:
                At = np.vstack((At, A))
                bt = np.vstack((bt, b))
            x = self.calc_feature(A, b)
            try:
                jacobian = self.calc_jacobian_of_measurement(x)
            except ZeroDivisionError:
                del measurement
                continue
            jac_list.append(jacobian)
            i += 1
        for g in jac_list:
            covariance += g.T.dot(g)
        covariance = np.square(constants.sigma_meas)*covariance
        return x.reshape(3, 1), covariance

    def calc_feature(self, A, b):
        """
        Calculate the feature
        Args:
            A, b: matrix for the last measurements (Eq. 5.42)
        """
        try:
            x = inv(A.T.dot(A)).dot(A.T).dot(b)
        except np.linalg.linalg.LinAlgError:
            print "Determinant is equal zero"
            x = np.array([0, 0, 0]).reshape(3, 1)
        return x

    def calc_matrix_Ab_parameters(self, img_coord, pose):
        """
        Calculate Ab parameters (Eq. 5.42)
        Args:
            img_coord: measurements SIFT image coordination [u,v]
            pose: pose object
        """
        #Rci = rotation_matrix(pose)
        #Rmi = rotation_matrix(pose)
        Rci = np.eye(3)
        Rmi = np.eye(3)
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


    def calc_mean(self):
        """
        Calculate mean
        Args:
            mean: [mx my mz] - 3D feature coordinates list
        """
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

    def calc_covariance(self):
        """
        Calculate covariance
        Returns:
            covariance: Covariance matrix 3x3
        """
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

def calc_distance_two_descriptors(d1, d2):
    """
    Calculate the distance between two 128-demensional sift features descriptors
    """
    dist = np.linalg.norm(d1 - d2)
    return dist

def rotation_matrix_mi(pose):
    """
    Form the rotation matrix that convert camera to map frame(C/M)
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

def rotation_matrix_ci(pose):
    """
    Form the rotation matrix that convert camera to map frame(C/M)
    Args:
        pose: particle`s euler_angles array [psi theta phi]
    """
    psi = pose.euler_angles[0]
    theta = pose.euler_angles[1]
    phi = pose.euler_angles[2]

    Rci = np.array([[cos(psi)*cos(theta), sin(psi)*cos(theta), cos(phi)*sin(theta)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*sin(phi)],
                   [cos(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), sin(phi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(theta)*cos(phi)]])
    return Rci

def rotation_matrix(pose):
    """
    Form the rotation matrix that convert camera to map frame(C/M)
    Args:
        pose: particle`s euler_angles array [psi theta phi]
    """
    psi = pose.euler_angles[0]
    theta = pose.euler_angles[1]
    phi = pose.euler_angles[2]

    r11 = cos(theta)*cos(phi)
    r21 = cos(theta)*sin(phi)
    r31 = (-1)*sin(theta)
    r12 = sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)
    r22 = sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi)
    r32 = sin(phi)*cos(theta)
    r13 = cos(phi)*cos(theta)*cos(phi) + sin(psi)*sin(phi)
    r23 = sin(psi)*cos(theta)
    r33 = cos(psi)*cos(theta)

    R = np.array([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])
    return R