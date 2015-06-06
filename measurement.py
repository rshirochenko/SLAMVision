import cv2
import numpy as np
import math
from collections import OrderedDict


class Measurement_J(object):
    def __init__(self, point, descriptor):
        self.point = point
        self.descriptor = descriptor


class Measurement_K(object):
    def __init__(self, point, descriptor, last_observed):
        self.point = point
        self.descriptor = descriptor
        self.last_observed = last_observed


class Measurement(object):
    n = 5  # number of best sift features
    mdist = 0.6  # the euclidean distance ration threshold (determined empirically)
    mdotprod = 0.8  # the additional logical check (determined empirically) Eq.2.3
    pixel_cutoff = 15  # the parameter for distance between sif points

    def __init__(self, table_K):
        """
        :type table_K: SIFT feature points reference table K
        (contains point, 128-vector descriptor, last observed)
        """
        self.table_K = table_K  # SIFT data structure
        self.sift = cv2.SIFT(self.n)

    def init_measurement(self, img):
        kpoint, des = self.calc_sift_points(img)
        table_K = self.create_table_K(kpoint, des)
        self.table_K = table_K

    def make_measurement(self, img):
        kpoint, des = self.calc_sift_points(img)
        table_J = self.create_table_J(kpoint, des)
        self.update_table_K(table_J)
        current_points = self.get_current_points()
        return table_J, current_points

    def get_current_points(self):
        current_points = {}
        for key in self.table_K:
            if self.table_K[key].last_observed == 0:
                current_points[key] = self.table_K[key]
        return current_points

    """ Get the SIFT point from one measurement
    Args: img - input image(must be gray) """
    def calc_sift_points(self, img):
        kpoint, des = self.sift.detectAndCompute(img, None)
        return kpoint, des

    """ Creates the table J for saving SIFT features """
    def create_table_J(self, point, des):
        table_J = OrderedDict()
        i = 0
        for p in point:
            table_J[i] = Measurement_J(p.pt, des[i])
            i += 1
        return table_J

    """ Create the table K for saving SIFT features """
    def create_table_K(self, point, des):
        table_K = OrderedDict()
        i = 0
        for p in point:
            table_K[i] = Measurement_K(p.pt, des[i], 0)
            i += 1
        return table_K

    """ The classical correspondence algorithm """
    def update_table_K(self, table_J):
        table_K = self.table_K
        # Add +1(last_observed) for all measurements in table K and delete if last_observed > 6
        for feature_k in table_K:
            table_K[feature_k].last_observed += 1
            if table_K[feature_k].last_observed > 7:
                del table_K[feature_k]

        # Searching for features with the same descriptor
        for feature_j in table_J:
            i = 0
            # List with features from table K with smallest distance to current feature
            distance_holder = ([[0, 0], [0, 0]])
            for feature_k in table_K:
                desc_distance = self.calc_distance_two_descriptors(table_J[feature_j], table_K[feature_k])
                # Searching for two features with smallest distance
                if i < 2:
                    distance_holder[i] = (feature_k, desc_distance)
                else:
                    if i == 2:
                        if distance_holder[0][1] > distance_holder[1][1]:
                            distance_holder[1], distance_holder[0] = distance_holder[0], distance_holder[1]
                    if desc_distance < distance_holder[1][1]:
                        if desc_distance < distance_holder[0][1]:
                            distance_holder[0], distance_holder[1] = (feature_k, desc_distance), distance_holder[0]
                        else:
                            distance_holder[1] = (feature_k, desc_distance)
                i += 1
                # if ratio less than a threshold mdist that update descriptor form feature j to feature k
            #print "distance_holder", distance_holder
            if self.check_pixel_distance(distance_holder) and self.compare_with_threshold_mdist(distance_holder):
                feature_number = distance_holder[0][0]
                table_K[feature_number] = Measurement_K(table_J[feature_j].point, table_J[feature_j].descriptor, 0)
                #TODO: implement logical_check function
            else:
                table_K[next(reversed(table_K)) + 1] = Measurement_K(table_J[feature_j].point, table_J[feature_j].descriptor, 0)

    """ Calculates the distance between two 128-demensional sift features descriptors """
    def calc_distance_two_descriptors(self, d1, d2):
        dist = np.linalg.norm(d1.descriptor-d2.descriptor)
        return dist

    """ Checking for the threshold mdist(eq 2.1)
    Args: two_features = list[(feature_point,distance1),(feature_point,distance2)]
    Output: boolean 1/0 """
    def compare_with_threshold_mdist(self, two_features):
        distk1 = two_features[0][1]
        distk2 = two_features[1][1]
        if distk1/distk2 <= self.mdist:
            return 1

    """ Check for the pixel distance cutoff
    Args two_features = list[(feature_point,distance1),(feature_point,distance2)]
    pixel_cutoff - parameter choosing empiricaly(by default 15)
    Output: boolean 1/0 """
    def check_pixel_distance(self, two_features):
        p1 = self.table_K[two_features[0][0]].point
        p2 = self.table_K[two_features[1][0]].point
        dist = math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        if dist < 15:
            return 1

    """ Logical check comparing with mdotprod
    pixel_cutoff - parameter choosing empiricaly(by default 15)
    Output: boolean 1/0 """
    def logical_check(self, two_features, distj):
        distk1 = self.table_K[two_features[0][0]].descriptor
        res = np.dot(distj, distk1)
        print 'logical check', res/1000000
        return 1


def change_keys(d):
    if type(d) is dict:
        return dict([(k+'abc', change_keys(v)) for k, v in d.items()])
    else:
        return d


def show_res(tab):
    for key in tab:
        print key, ' point', tab[key].point, ' last_observed', tab[key].last_observed


def show_table_J(tab):
    for key in tab:
        print key, ' point', tab[key].point


def swap(s1, s2):
    return s2, s1

"""
def main():
    K = []
    meas = Measurement(K)
    meas.init_measurement(gray1)
    show_res(meas.table_K)

    print "Second measurement"
    meas.make_measurement(gray2)
    show_res(meas.table_K)

    print "Third measurement"
    meas.make_measurement(gray3)
    show_res(meas.table_K)

if  __name__ =='__main__':main()
"""

