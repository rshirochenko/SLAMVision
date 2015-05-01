import camera
from initialization import *
from motion_model import *
import pose
import initialization
from math import cos, sin


class FastSLAM(object):
    fc = 0.1
    Rprop = []
    P = [] #TODO: proposal distribution (eq. 5.20)

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

    #TODO: optimize the feature calculation in the camera frame using eq.5.4
    def calc_particle_distribution(self, particle, Z_table_K):
        X_map = particle.X_map
        covariance_sum = 0
        mean_part = 0
        for j in X_map:
            if j in Z_table_K:
                mean = X_map[j].mean
                X_c = self.convert_feature_to_camera_frame(particle.pose, mean)
                measurement_predicted = self.measurement_model(X_c)
                Gx = self.calc_jacobian_Gx(X_c)
                Gs = self.calc_jacobian_Gs(X_c, particle.pose)
                Q = self.calc_Q(Gx,X_map[j].covariance)
                #TODO: check the equetion 5.20 and 5.21. Found problem with dimensionality
                covariance_sum = covariance_sum + inv(Gx.T.dot(inv(Q)).dot(Gx))
                measurement_current = np.asarray(Z_table_K[j].point).reshape(2,1)
                mean_part = Gx.T.dot(inv(Q)).dot(measurement_current-measurement_predicted)
        mean = covariance_sum.dot(mean_part) + particle.pose.position
        return mean

    def measurement_update(self, particle, Z_table_K):
        X_map = particle.X_map
        for j in X_map:
            if j in Z_table_K:
                mean = X_map[j].mean
                covariance = X_map[j].covariance
                X_c = self.convert_feature_to_camera_frame(particle.pose, mean)
                measurement_predicted = self.measurement_model(X_c)
                measurement_current = np.asarray(Z_table_K[j].point).reshape(2,1)
                Gx = self.calc_jacobian_Gx(X_c)
                Gs = self.calc_jacobian_Gs(X_c, particle.pose)

                #EKF measurement update
                mean_updated, covariance_updated = self.EKF_measurement_update(Gx,Gs,mean,covariance, measurement_current, measurement_predicted)
                X_map[j].mean = mean_updated
                X_map[j].covariance = covariance_updated
        return measurement_predicted

    def EKF_measurement_update(self, Gx, Gs, mean, covariance,measurement_current, measurement_predicted):
        P = np.identity(3) #TODO: change P here
        sigma = 0.1 #TODO:change sigma here
        Q = Gx.dot(P).dot(Gx.T) + Gx.dot(covariance).dot(Gx.T) + sigma*np.eye(2)
        K = covariance.dot(Gx.T).dot(inv(Q))
        mean = mean + K.dot(measurement_current-measurement_predicted)
        covariance = (np.eye(3) - K.dot(Gx)).dot(covariance)
        return mean, covariance


    def convert_feature_to_camera_frame(self, pose, mean):
        position = pose.position
        Rcm = self.form_matrix_Rcm(pose)
        X_c = Rcm.dot(mean) + position
        return X_c

    def measurement_model(self,Xc):
        measurement = (self.fc/Xc[2])*Xc[:2]
        return measurement

    def calc_jacobian_Gs(self, feature, pose):
        x = float(feature[0])
        y = float(feature[1])
        z = float(feature[2])
        px = float(pose.position[0])
        py = float(pose.position[1])
        pz = float(pose.position[2])
        fc = self.fc
        dg_by_dx = np.matrix([[fc/z, 0, ((-1)*fc*x)/(z*z)],
                              [0, fc/(z*z), ((-1)*fc*y)/(z*z)]])
        dx_by_ds = np.matrix([[(-1)*y+px, z - pz, 0, 1, 0, 0],
                              [x - px, 0, -z +pz, 0, 1, 0],
                              [0, -x + px, y - py, 0, 0, 1]])
        Gs = dg_by_dx*dx_by_ds
        return Gs

    def calc_jacobian_Gx(self, feature):
        x = float(feature[0])
        y = float(feature[1])
        z = float(feature[2])
        fc = self.fc
        dg_by_dx = np.array([[fc/z, 0, ((-1)*fc*x)/(z*z)],
                             [0, fc/(z*z), ((-1)*fc*y)/(z*z)]])
        Gx = dg_by_dx
        return Gx

    def form_matrix_Rcm(self, pose):
        euler_angles = pose.euler_angles
        psi = euler_angles[0]
        theta = euler_angles[1]
        phi = euler_angles[2]
        r11 = cos(psi)*sin(theta)
        r21 = sin(psi)*cos(theta)
        r31 = (-1)*sin(theta)

        r12 = cos(psi)*sin(theta)*sin(phi)-sin(psi)*cos(phi)
        r22 = sin(psi)*sin(theta)*sin(phi)+cos(psi)*cos(phi)
        r32 = cos(theta)*sin(phi)

        r13 = cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)
        r23 = sin(psi)*sin(theta)*cos(phi)-cos(psi)*sin(phi)
        r33 = cos(theta)*cos(phi)

        Rci = np.array([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
        return Rci

    #TODO: Add Rprop
    def calc_Q(self, Gx, covariance):
        Q = Gx.dot(covariance).dot(Gx.T)
        return Q



def main():
    print("Hello")
    fastslam = FastSLAM()


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

    for particle_id in particles_dict:
        print(fastslam.measurement_update(particles_dict[particle_id], Z_table_K))

if  __name__ =='__main__':main()
