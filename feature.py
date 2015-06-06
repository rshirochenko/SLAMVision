import math
import numpy as np
import constants
from numpy.linalg import inv


class Feature(object):
    def __init__(self, mean, covariance, descriptor):
        self.mean = mean
        self.covariance = covariance
        self.descriptor = descriptor


class XMapDictionary(object):
    X_map_dict = []

    def update_X_map_dictionary(self, features_temp, current_measurements):
        features_temp.get_measurements(current_measurements, self.X_map_dict)

    def add_to_X_map_dict(self, feature):
        X_map_dict = self.X_map_dict
        for X_map_feature in X_map_dict:
            if calc_distance_two_descriptors(X_map_feature.descriptor, feature.desctiptor) < 100:
                return
            else:
                X_map_dict.append(feature)

class FeaturesTemp(object):
    measurement_dict = {}

    def get_measurements(self, current_measurements, X_map_dict):
        measurement_dict = self.measurement_dict
        # Check the measurement_dict for emptiness
        if not bool(measurement_dict):
            for key in current_measurements:
                measurement_dict[key] = [current_measurements[key]]
            return
        # Add observed measurements to feature temp storage
        for key in current_measurements:
            if self.check_Xmap(X_map_dict):
                break
            else:
                self.check_add_to_measurement_dict(current_measurements[key])
                self.check_add_to_Xmap(X_map_dict)
        for meas in measurement_dict[3]:
            print "measurements_dict", meas.point

    def check_add_to_measurement_dict(self, current_measurements):
        measurement_dict = self.measurement_dict
        for j in measurement_dict:
            if calc_distance_two_descriptors(current_measurements.descriptor,
                                                      measurement_dict[j][-1].descriptor) < 100:
                measurement_dict[j].append(current_measurements)

    def check_Xmap(self, X_map_dict):
        measurement_dict = self.measurement_dict
        for feature in X_map_dict:
            for j in self.measurement_dict:
                if calc_distance_two_descriptors(feature.descriptor,
                                                      measurement_dict[j][-1].descriptor) < 100:
                    return 1

    def check_add_to_Xmap(self, X_map_dict):
        measurement_dict = self.measurement_dict
        for j in self.measurement_dict:
            if self.check_for_views_number(measurement_dict[j]):
                mean, covariance = self.EKF_initialization(measurement_dict[j])
                print "mean", mean, " covariance", covariance
                X_map_dict.append(measurement_dict[j])
                del measurement_dict[j]

    def check_for_views_number(self, measurement_dict_element):
        if len(measurement_dict_element) == constants.feature_time_for_init:
            return 1

    def EKF_initialization(self, measurement_dict_element):
        x = 0
        covariance = 0
        i = 1
        At = np.array([])
        bt = np.array([])
        jac_sum = 0
        for measurement in measurement_dict_element:
            A, b = self.calc_matrix_Ab_parameters(measurement.point)
            if i == 1:
                At = A
                bt = b
            else:
                At = np.vstack((At, A))
                bt = np.vstack((bt, b))
            #print "i", i
            #print "At", At
            #print
            #print "bt", bt
            try:
                x = self.calc_feature(At, bt)
                jacobian = self.calc_jacobian_of_measurement(x)
                jac_sum = jac_sum + jacobian.T.dot(jacobian)
                # TODO:check the covariance calculation (eq.5.44)
                covariance = jac_sum
            except:
                print "Oops!  That was no valid number.  Try again..."
                break
            i += 1
        return x, covariance

    """Args: A, b matrix for the last measurements
    """

    def calc_feature(self, A, b):
        try:
            x = inv(A.T.dot(A)).dot(A.T).dot(b)
        except np.linalg.linalg.LinAlgError:
            print "Determinant is equal zero"
            x = 0
        return x


    """Args: img_coord - 1 measurements SIFT image coordination [u,v]
    """

    def calc_matrix_Ab_parameters(self, img_coord):
        Rci = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        Rmi = np.array([[11, 12, 13],
                        [14, 15, 16],
                        [17, 18, 19]])
        p = np.array([3, 3, 3])
        fc = constants.fc
        # img_coord = [10,10] #(u,v)

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

    """Args: List[x[x y z]] - 3D feature coordinates list
    """

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
    Output: Covariance matrix 3x3
    """

    def calc_covariance(self):
        std = self.calc_std_square()
        jac_sum = 0
        for feature in self.features_list:
            jacobian = self.calc_jacobian_of_measurement(feature)
            jac_sum = jac_sum + jacobian.T * jacobian
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
def calc_distance_two_descriptors(self, d1, d2):
    dist = np.linalg.norm(d1 - d2)
    return dist