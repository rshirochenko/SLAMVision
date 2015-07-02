import cv2
import numpy as np
import math
from collections import namedtuple

class Measurement(object):
    n = 3 #number of best sift features
    mdist = 1 # the euclidean distance ration threshold (determined empirically)
    mdotprod = 0.8 # the additional logical check (determined empirically) Eq.2.3

    def __init__(self, table_K):
        self.table_K = table_K # SIFT data structure
        self.sift = cv2.SIFT(self.n)

    def init_measurement(self,img):
        kpoint, des = self.calc_sift_points(img)
        table_K = self.create_table_K(kpoint, des)
        for point in kpoint:
            print "init_measurement",point
            print point.pt
        self.table_K = table_K

    def make_measurement(self, img):
        kpoint, des = self.calc_sift_points(img)
        table_J = self.create_table_J(kpoint, des)
        for point in kpoint:
            print "make_measurement",point
            print point.pt
        self.update_t_K(table_J)


    """get the SIFT point from one measurement
    Args: img - input image(must be gray)"""
    def calc_sift_points(self, img):
        kpoint, des = self.sift.detectAndCompute(img,None)
        return kpoint, des

    """Creates the table J for saving SIFT features"""
    def create_table_J(self,point, des):
        table_J = dict()
        i = 0
        for p in point:
            table_J[p] = des[i]
            i = i + 1
        return table_J

    """Creates the table K for saving SIFT features"""
    def create_table_K(self, point, des):
        table_K = dict()
        i = 0
        for p in point:
            table_K[p] = [des[i],0]
            i = i + 1
        return table_K

    """Updates the SIFT features table.
    Thereshold value. If the feature point do not not emerge in the 8 frames, then delete in from the table"""
    def update_table_K(table, point, des, threshold):
        i = 0
        for x, y in table.items():
            y[1] = y[1] + 1
            if y[1] > threshold:
                del table[x]
        for p in point:
            table[p] = [des[i],0]
            i = i + 1

    """The classical correspondence algorithm"""
    def update_t_K(self, table_J):
        table_K = self.table_K
        for feature_j in table_J:
            i = 0
            distance_holder = [(0,0),(0,0)] # list with features from table K with smallest distance to current feature
            for feature_k in table_K:
                desc_distance = self.calc_distance_two_descriptors(table_J[feature_j], table_K[feature_k][0])

                # Searching for two features with smallest distance
                if i < 2:
                    distance_holder[i] = (feature_k, desc_distance)
                else:
                    if i == 2:
                        if distance_holder[0][1] > distance_holder[1][1]:
                            temp = distance_holder[0]
                            distance_holder[1] = distance_holder[0]
                            distance_holder[0] = temp
                    if desc_distance < distance_holder[1][1]:
                        if desc_distance < distance_holder[0][1]:
                            temp = distance_holder[0]
                            distance_holder[0] = (feature_k,desc_distance)
                            distance_holder[1] = temp
                        else:
                            distance_holder[1] = (feature_k, desc_distance)
                i += 1
            # if ratio less than a threshold mdist that update descriptor form feature j to feature k
            #print distance_holder
            if self.compare_with_threshold_mdist(distance_holder):
                print "Yes!"
                #table_K[distance_holder[0][0]] = [table_J[feature_j], 0]
                #table_K[feature_j] = table_K[distance_holder[0][0]]
                #del dict[distance_holder[0][0]]
                #TODO: write the Eq.2.3 logical check. Now dot product gives to big result
                #if np.dot(table_J[feature_j].T, table_K[distance_holder[0][0]][0]) < self.mdotprod:
                #    print("Yes!")




    def calc_distance_two_points(p1, p2):
        dist = math.sqrt((p1.x-p2.x)^2 + (p1.y-p2.y)^2)
        return dist


    def compare_two_image_points(self,p1,p2):
        dist = math.sqrt((p1.x-p2.x)^2 + (p1.y-p2.y)^2)
        if(dist < 15):
            return 0

    """Calculates the distance between two 128-demensional sift features descriptors"""
    def calc_distance_two_descriptors(self, d1, d2):
        dist = np.linalg.norm(d1-d2)
        return dist

    """Checking for the threshold mdist(eq 2.1)
    Args: two_features = list[(feature_point,distance1),(feature_point,distance2)]
    Output: boolean 1/0"""
    def compare_with_threshold_mdist(self,two_features):
        distk1 = two_features[0][1]
        distk2 = two_features[1][1]
        if (distk2/distk1 <= self.mdist):
            return 1

def change_keys(d):
  if type(d) is dict:
    return dict([(k+'abc', change_keys(v)) for k, v in d.items()])
  else:
    return d



"""
def main():
    print("Hello")
    img1 = cv2.imread('seqtest0001.pgm')
    img2 = cv2.imread('seqtest0002.pgm')
    img3 = cv2.imread('seqtest0003.pgm')
    img4 = cv2.imread('seqtest0004.pgm')
    img5 = cv2.imread('seqtest0005.pgm')
    img6 = cv2.imread('seqtest0006.pgm')


    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

    K = []
    meas = Measurement(K)
    meas.init_measurement(gray1)
    meas.make_measurement(gray2)
    #meas.make_measurement(gray3)

if  __name__ =='__main__':main()

"""
