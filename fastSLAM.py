import camera
from initialization import *
from motion_model import *
import pose
import initialization


class FastSLAM(object):
    fc = 0.1

    def __init__(self):
        print("MonoSLAM is initiated")

    def time_update(self):
        pose_current = pose.Pose([10,20,13],[1,1,2],[10,20,30])
        return

    def update_stage(self, previous_pose, features, particles):
        motion_model = Motion_model()
        X = dict()
        for particle in particles:
            current_pose_predicted = motion_model.calcNextPose(previous_pose)
            for feature in features:
                if(feature in X.keys):
                    mean = self.calcMean()
                    jacobian = self.calcJacobian()
                    covariance = self.calcCovariance()
                    weight = 1 #default value
                else:
                    measurement_prediction = self.calcMeasurement()
                    jacobian = self.calcJacobian()
                    measurement_covariance = self.calcCovariance()
                    kalman_gain = self.calcKalmanGain()
                    #update mean, update covariance,
                    #calc the weight

    def calc_covariance_s(self, pose):
        return

    def calc_jacobian_s(self, feature, pose):
        x = feature.point[0]
        y = feature.point[1]
        z = feature.point[2]
        px = pose.position[0]
        py = pose.position[1]
        pz = pose.position[2]
        fc = self.fc
        dg_by_dx = np.matrix([[fc/z, 0, ((-1)*fc*x)/(z*z)],
                              [0, fc/(z*z), ((-1)*fc*y)/(z*z)]])
        dx_by_ds = np.matrix([[(-1)*y+px, z - pz, 0, 1, 0, 0],
                              [x - px, 0, -z +pz, 0, 1, 0],
                              [0, -x + px, y - py, 0, 0, 1]])
        Gs = dg_by_dx*dx_by_ds
        return Gs

    def calc_jacobian_x(self, feature):
        x = feature.point[0]
        y = feature.point[1]
        z = feature.point[2]
        fc = self.fc
        dg_by_dx = np.matrix([[fc/z, 0, ((-1)*fc*x)/(z*z)],
                              [0, fc/(z*z), ((-1)*fc*y)/(z*z)]])
        Gx = dg_by_dx
        return Gx



def main():
    print("Hello")
    monoslam = FastSLAM()

    test_data()

    #1.Initiate the camera and set camera parametrs
    cam = camera.Camera()
    cam.setCameraParametrs()

    #2. Initiate the motion model
    initialize = initialization.Initialization()
    a, Z_table_K = initialize.get_img_coordinates_Z()

    particles_dict = pose.makeParticlesDict(2)

    motion_model = Motion_model()
    motion_model.rotational_motion_model(particles_dict)
    motion_model.translational_optimization(particles_dict,Z_table_K)

    #3. Initiate the measurement model

    #Determine the constant variables
    number_of_visible_features = 0


if  __name__ =='__main__':main()
