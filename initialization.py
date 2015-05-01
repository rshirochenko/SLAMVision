import numpy as np
from numpy.linalg import inv
import feature
import measurement
import cv2
from collections import OrderedDict

class Initialization(object):
    features_list = []
    fc = 0.1

    def __init__(self):
        self.X_map = self.build_X_map()

    def build_X_map(self):
        points_Z_point_only, points_Z = self.get_img_coordinates_Z()
        X_map = OrderedDict() # X map with 3D coordinates features [x y z]
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
                    At = np.vstack((At,A))
                    bt = np.vstack((bt,b))
                i += 1
            x = self.calc_feature(At, bt)
            jacobian = self.calc_jacobian_of_measurement(x)
            jac_sum = jac_sum + jacobian*jacobian.T
            covariance = inv(jac_sum)
            X_map[point] = feature.Feature(x,x,covariance)
        return X_map

    def get_img_coordinates_Z(self):
        #receive the image and calc SIFT measurements TODO: make it autonomous
        #test images. TODO: make it autonomous
        img1 = cv2.imread('seqtest0001.pgm')
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        #initial stage
        K = []
        points_Z_init = OrderedDict()
        measurements = measurement.Measurement(K)
        measurements.init_measurement(gray1)
        points_Z = measurements.table_K

        #starts to calculate image points
        #TODO: form the coordinates for each measurements should be 4 coordinates. If empty, put (0,0) coordinates
        for i in range(2,5):
            imgname = "seqtest000" + str(i) + ".pgm"
            img = cv2.imread(str(imgname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            measurements.make_measurement(gray)
            points_Z = measurements.table_K
            for key in points_Z:
                if key in points_Z_init: points_Z_init[key].append(points_Z[key].point)
                else: points_Z_init[key] = [points_Z[key].point]
        return points_Z_init, points_Z

    """Args: A, b matrix for the last measurements
    """
    def calc_feature(self, A, b):
        x = inv(A.T*A)*A.T*b
        return x

    """Args: img_coord - 1 measurements SIFT image coordination [u,v]
    """
    def calc_matrix_Ab_parameters(self, img_coord):
        Rci = np.matrix([[1,2,3],
                        [4,5,6],
                        [7,8,9]])
        Rmi = np.matrix([[11,12,13],
                         [14,15,16],
                         [17,18,19]])
        p = np.matrix([3,3,3])
        fc = self.fc
        #img_coord = [10,10] #(u,v)

        a = Rci.item((0,0))*Rmi.item((0,0)) + Rci.item((0,1))*Rmi.item((0,1)) + Rci.item((0,2))*Rmi.item((0,2))
        b = Rci.item((0,0))*Rmi.item((1,0)) + Rci.item((0,1))*Rmi.item((1,1)) + Rci.item((0,2))*Rmi.item((1,2))
        c = Rci.item((0,0))*Rmi.item((2,0)) + Rci.item((0,1))*Rmi.item((2,1)) + Rci.item((0,2))*Rmi.item((2,2))
        d = Rci.item((0,0))*p.item(0) + Rci.item((0,1))*p.item(1) + Rci.item((0,2))*p.item(2)
        e = Rci.item((1,0))*Rmi.item((0,0)) + Rci.item((1,1))*Rmi.item((0,1)) + Rci.item((1,2))*Rmi.item((0,2))
        f = Rci.item((1,0))*Rmi.item((1,0)) + Rci.item((1,1))*Rmi.item((1,1)) + Rci.item((1,2))*Rmi.item((1,2))
        g = Rci.item((1,0))*Rmi.item((2,0)) + Rci.item((1,1))*Rmi.item((2,1)) + Rci.item((1,2))*Rmi.item((2,2))
        h = Rci.item((1,0))*p.item(0) + Rci.item((1,1))*p.item(1) + Rci.item((1,2))*p.item(2)
        l = Rci.item((2,0))*Rmi.item((0,0)) + Rci.item((2,1))*Rmi.item((0,1)) + Rci.item((2,2))*Rmi.item((0,2))
        m = Rci.item((2,0))*Rmi.item((1,0)) + Rci.item((2,1))*Rmi.item((1,1)) + Rci.item((2,2))*Rmi.item((1,2))
        n = Rci.item((2,0))*Rmi.item((2,0)) + Rci.item((2,1))*Rmi.item((2,1)) + Rci.item((2,2))*Rmi.item((2,2))
        o = Rci.item((2,0))*p.item(0) + Rci.item((2,1))*p.item(1) + Rci.item((2,2))*p.item(2)

        A = np.matrix([[fc*a - img_coord[0]*l, fc*b - img_coord[0]*m, fc*c - img_coord[0]*n],
                       [fc*e - img_coord[1]*l, fc*f - img_coord[1]*m, fc*g - img_coord[1]*n]])

        b = np.matrix([[img_coord[0]*o - fc*d],
                      [img_coord[1]*o - fc*h]])
        return A,b


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
        mean = np.array([x,y,z])/number_of_features
        return mean

    """Args: x[x y z]] - 3D feature coordinate
    Output: Covariance matrix 3x3
    """
    def calc_covariance(self):
        std = self.calc_std_square()
        jac_sum = 0
        for feature in self.features_list:
            jacobian = self.calc_jacobian_of_measurement(feature)
            jac_sum = jac_sum + jacobian*jacobian.T
        covariance = std*inv(jac_sum)
        return covariance

    def calc_std_square(self):
        std = self.features_list[3].std()
        return np.square(std)

    def calc_jacobian_of_measurement(self,feature):
        feature = np.array(feature).reshape(-1,).tolist()
        fc = self.fc
        g11 = fc/feature[2]
        g12 = 0
        g13 = (-1)*fc/(feature[2]*feature[2])
        g21 = 0
        g22 = fc/feature[2]
        g23 = (-1)*float(feature[1])/(feature[2]*feature[2])
        jacobian = np.matrix([[g11,g12,g13],
                              [g21,g22,g23]])

        return jacobian

def main():
    print("Hello")

    init = Initialization()

    X_map =  init.X_map

    for keys in X_map:
        print X_map[keys].point


if  __name__ =='__main__':main()
