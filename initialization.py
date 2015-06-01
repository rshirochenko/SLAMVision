import numpy as np
from numpy.linalg import inv
import feature
import measurement
import cv2
import math
from collections import OrderedDict


class Initialization(object):
    features_list = []
    img_coordinates_dict = {}
    fc = 0.1
    counter = 0

    def get_img_coordinates(self, measurements):
        self.counter += 1
        for key in measurements:
            if self.counter == 1:
                if key not in self.img_coordinates_dict:
                    self.img_coordinates_dict[key] = [measurements[key].point]
            if self.counter > 1:
                for i in self.img_coordinates_dict:
                    if check_pixel_distance(measurements[key].point, self.img_coordinates_dict[i][-1]):
                        self.img_coordinates_dict[i].append(measurements[key].point)
                    if (self.counter - 1) <= len(self.img_coordinates_dict[i]) < self.counter:
                        self.img_coordinates_dict[i].append((0, 0))


    def build_X_map(self):
        points_Z_point_only = self.img_coordinates_dict
        X_map = OrderedDict()  # X map with 3D coordinates features [x y z]
        for point in points_Z_point_only:
            At = np.matrix([])
            bt = np.matrix([])
            i = 0
            jac_sum = 0
            for img_coord in points_Z_point_only[point]:
                A, b = self.calc_matrix_Ab_parameters(img_coord)
                if i == 0:
                    At = A
                    bt = b
                else:
                    At = np.vstack((At, A))
                    bt = np.vstack((bt, b))
                i += 1
            x = self.calc_feature(At, bt)
            jacobian = self.calc_jacobian_of_measurement(x)
            jac_sum = jac_sum + jacobian.T.dot(jacobian)
            # TODO:check the covariance calculation (eq.5.44)
            covariance = jac_sum
            X_map[point] = feature.Feature(x, covariance)
        return X_map

    """Args: A, b matrix for the last measurements
    """
    def calc_feature(self, A, b):
        x = inv(A.T.dot(A)).dot(A.T).dot(b)
        return x

    """Args: img_coord - 1 measurements SIFT image coordination [u,v]
    """
    def calc_matrix_Ab_parameters(self, img_coord):
        Rci = np.matrix([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        Rmi = np.matrix([[11, 12, 13],
                         [14, 15, 16],
                         [17, 18, 19]])
        p = np.matrix([3, 3, 3])
        fc = self.fc
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
        fc = self.fc
        g11 = fc / feature[2]
        g12 = 0
        g13 = (-1) * fc / (feature[2] * feature[2])
        g21 = 0
        g22 = fc / feature[2]
        g23 = (-1) * float(feature[1]) / (feature[2] * feature[2])
        jacobian = np.matrix([[g11, g12, g13],
                              [g21, g22, g23]])

        return jacobian

def check_pixel_distance(p1, p2):
        dist = math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        if dist < 15:
            return 1

def main():
    initial = Initialization()
    points_z = initial.get_img_coordinates_Z()
    print(points_z)


if __name__ == '__main__': main()