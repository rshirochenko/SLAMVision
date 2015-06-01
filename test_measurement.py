import unittest
from measurement import *
from video import *
import cv2

class MeasurementsTestCase(unittest.TestCase):
    """Tests for measurements.py
    The test will be based on the video file 4_film1.wmv
    For the test will be calculated 5 sift feature points
    """

    """
    def test_init_measurement(self):
        # Open the frame #4 because from this frame the number of sift points is stable
        ret, cap = open_video(path)
        ret, frame = get_frame_n(ret, cap, 4)

        K = []
        meas = Measurement(K)
        meas.init_measurement(frame)

        sift = cv2.SIFT(4)
        kp, des = sift.detectAndCompute(frame, None)

        check = True
        for keypoint in kp:
            for key in meas.table_K:
                if keypoint.pt == meas.table_K[key].point:
                    check = True
                else:
                    check = False
        self.assertTrue(check)
    """

    def test_create_table_J(self):
        ret, cap = open_video(path)
        ret, frame = get_frame_n(ret, cap, 4)

        K = []
        meas = Measurement(K)
        meas.init_measurement(frame)

        kpoint, des = meas.calc_sift_points(frame)
        table_J = meas.create_table_J(kpoint, des)
        #print  table_J

        self.assertTrue(True)

    def test_update_table_K(self):
        ret, cap = open_video(path)
        ret, frame = get_frame_n(ret, cap, 5)
        K = []
        meas = Measurement(K)
        meas.init_measurement(frame)
        print "initialization5"
        show_res(meas.table_K)

        for i in range(6, 10):
            print "frame ", i
            ret, frame = get_frame_n(ret, cap, i)
            kpoint, des = meas.calc_sift_points(frame)
            table_J = meas.create_table_J(kpoint, des)
            print "table_J", i, show_table_J(table_J)
            meas.update_table_K(table_J)
            print "table_K", i, show_res(meas.table_K)

        self.assertTrue(True)

    def test_compare_with_threshold_mdist(self):
        ret, cap = open_video(path)
        ret, frame = get_frame_n(ret, cap, 4)

        K = []
        meas = Measurement(K)
        meas.init_measurement(frame)

        ret, frame = get_frame_n(ret, cap, 5)
        kpoint, des = meas.calc_sift_points(frame)
        des_frame_5 = des[0]

        ret, frame = get_frame_n(ret, cap, 5)
        kpoint, des = meas.calc_sift_points(frame)
        des_frame_6 = des[0]

        distance = np.linalg.norm(des_frame_5 - des_frame_6)
        print "Distance", distance

        self.assertTrue(True)



if __name__ == '__main__':
    unittest.main()
