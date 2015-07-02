import cv2
import numpy as np
import math
from collections import OrderedDict


class Measurement_J(object):
    """
    Element(row) of the table J
    Attributes:
        point: Coordinates of the SIFT measurement
        descriptor: 128-vector of SIFT measurement
    """
    def __init__(self, point, descriptor):
        self.point = point
        self.descriptor = descriptor


class Measurement_K(object):
    """
    Element(row) of the table K
    Attributes:
        point: Coordinates of the SIFT measurement
        descriptor: 128-dimension vector of SIFT measurement
        last_observed: Integer number - times since last observation
    """
    def __init__(self, point, descriptor, last_observed):
        self.point = point
        self.descriptor = descriptor
        self.last_observed = last_observed


class Measurements_Z(object):
    """
    Class describes the measurements(Z) for each time(frame) moment

    Attributes:
        table_K: Table K that contains the SIFT measurements(p.21 in Augenstein)
        sift: Using OpenCV for SIFT detecting. Declaring OpenCV object here.

    """
    n = 5  # number of best sift features
    mdist = 0.6  # the euclidean distance ration threshold (determined empirically)
    mdotprod = 0.8  # the additional logical check (determined empirically) Eq.2.3
    pixel_cutoff = 15  # the parameter for distance between sift points

    def __init__(self, table_K):
        """
        table_K: SIFT feature points reference table K
        (contains point, 128-vector descriptor, last observed)
        """
        self.table_K = table_K  # SIFT data structure
        self.sift = cv2.SIFT(self.n)


    def init_measurement(self, img):
        """
        The t=0 initial stage getting the Z(SIFT) measurements
        Args:
            img: Frame(image variable in gray color
        """
        kpoint, des = self.calc_sift_points(img)
        table_K = self.create_table_K(kpoint, des)
        self.table_K = table_K
        current_points = self.get_current_measurements()
        return current_points

    def make_measurement(self, img):
        """
        Estimate current frame SIFT measurements and update the table K
        Args:
            img: Frame(image) variable in gray color
        """
        kpoint, des = self.calc_sift_points(img)
        table_J = self.create_table_J(kpoint, des)
        self.update_table_K(table_J)
        current_measurements = self.get_current_measurements()
        return current_measurements

    def get_current_measurements(self):
        """
        Return the last observed measurements from table K with last_observed equal 0
        """
        current_measurements = {}
        for key in self.table_K:
            if self.table_K[key].last_observed == 0:
                current_measurements[key] = self.table_K[key]
        return current_measurements

    def calc_sift_points(self, img):
        """ Get the SIFT point from one measurement
            Args:
                img: Frame(image) variable in gray color
        """
        kpoint, des = self.sift.detectAndCompute(img, None)
        return kpoint, des

    def create_table_J(self, point, des):
        """
        Create the table J for saving SIFT features
        Args:
            point: Coordinates of the SIFT measurement
            descriptor: 128-vector of SIFT measurement
        """
        table_J = OrderedDict()
        i = 0
        for p in point:
            table_J[i] = Measurement_J(p.pt, des[i])
            i += 1
        return table_J

    def create_table_K(self, point, des):
        """
        Create the table K for saving SIFT features
        Args:
            point: Coordinates of the SIFT measurement
            descriptor: 128-dimension vector of SIFT measurement
            last_observed: Integer number - times since last observation
        """
        table_K = OrderedDict()
        i = 0
        for p in point:
            table_K[i] = Measurement_K(p.pt, des[i], 0)
            i += 1
        return table_K

    def update_table_K(self, table_J):
        """
        The classical correspondence algorithm between SIFTs from different frames
        """
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
            if self.check_pixel_distance(distance_holder) and self.compare_with_threshold_mdist(distance_holder):
                feature_number = distance_holder[0][0]
                table_K[feature_number] = Measurement_K(table_J[feature_j].point, table_J[feature_j].descriptor, 0)
                #TODO: implement logical_check function
            else:
                table_K[next(reversed(table_K)) + 1] = Measurement_K(table_J[feature_j].point, table_J[feature_j].descriptor, 0)

    def calc_distance_two_descriptors(self, d1, d2):
        """
        Calculate the distance between two 128-demensional sift features descriptors
        """
        dist = np.linalg.norm(d1.descriptor-d2.descriptor)
        return dist

    def compare_with_threshold_mdist(self, two_features):
        """
        Check for the threshold mdist(eq 2.1)
        Args:
            two_features = list[(feature_point,distance1),(feature_point,distance2)]
        Returns: boolean 1/0
        """
        distk1 = two_features[0][1]
        distk2 = two_features[1][1]
        if distk1/distk2 <= self.mdist:
            return 1

    def check_pixel_distance(self, two_features):
        """
        Check for the pixel distance cutoff
        Args:
            two_features: list[(feature_point,distance1),(feature_point,distance2)]
            pixel_cutoff: the distance between SIFTs coordinates (choosing empirical(by default 15))
        Attributes: boolean 1/0 """
        p1 = self.table_K[two_features[0][0]].point
        p2 = self.table_K[two_features[1][0]].point
        dist = math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        if dist < 15:
            return 1

    def logical_check(self, two_features, distj):
        """
        Logical check comparing with mdotprod threshold
        Args:
            pixel_cutoff: the distance between SIFTs coordinates (choosing empirical(by default 15))
        Attributes:
            boolean 1/0
        """
        distk1 = self.table_K[two_features[0][0]].descriptor
        res = np.dot(distj, distk1)
        print 'logical check', res/1000000
        return 1


def show_res(tab):
    """Print the table K contains"""
    for key in tab:
        print key, ' point', tab[key].point, ' last_observed', tab[key].last_observed

def show_table_J(tab):
    """Print the table J contains"""
    for key in tab:
        print key, ' point', tab[key].point

def swap(s1, s2):
    """ Simple swap function"""
    return s2, s1
