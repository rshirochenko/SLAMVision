from initialization import *
from motion_model import *
import pose
import initialization
from math import cos, sin, sqrt, exp, pi
import constants


class FastSLAM(object):
    fc = constants.fc
    Rprop = []
    P = [] #TODO: proposal distribution (eq. 5.20)
    M = constants.NUMBER_OF_PARTICLES  # total number of particles

    #TODO: optimize the feature calculation in the camera frame using eq.5.4
    """ FastSLAM 2.0 Proposal distribution for st
    Time update for st
    Args: particle - current particle from particle_filter(contains X_map_previous_step
            and predicted st)
         current_measurements -  measurements for the current time step
    """
    def calc_position_proposal_distribution(self, particle, current_measurements):
        X_map = particle.X_map_dict
        covariance_sum = 0
        mean_part = 0
        for x in X_map:
            if x.debug_key in current_measurements:
                mean = x.mean
                X_c = self.convert_feature_to_camera_frame(particle.pose, mean)
                measurement_predicted = self.measurement_model(X_c)
                Gx = self.calc_jacobian_Gx(X_c)
                Gs = self.calc_jacobian_Gs(X_c, particle.pose)
                Q = self.calc_Q(Gx, x.covariance)
                #TODO: check the equation 5.20 and 5.21. Found problem with dimensionality
                try:
                    covariance_sum += Gx.T.dot(inv(Q)).dot(Gx)
                    measurement_current = np.asarray(current_measurements[x.debug_key].point).reshape(2, 1)
                    mean_part += Gx.T.dot(inv(Q)).dot(measurement_current-measurement_predicted)
                except:
                    print "covariance_sum error"
                    pass
        #mean_st = covariance_sum.dot(mean_part) + particle.pose.coordinates
        mean_st = np.random.sample((3, 1))

    def measurement_update(self, particle, current_measurements):
        X_map = particle.X_map_dict
        weight_total = 1
        for j in X_map:
            if j in current_measurements:
                mean = X_map[j].mean
                covariance = X_map[j].covariance
                X_c = self.convert_feature_to_camera_frame(particle.pose, mean)
                measurement_predicted = self.measurement_model(X_c)
                measurement_current = np.asarray(current_measurements[j].point).reshape(2, 1)
                Gx = self.calc_jacobian_Gx(X_c)
                Gs = self.calc_jacobian_Gs(X_c, particle.pose)

                #EKF measurement update
                mean_updated, covariance_updated, Q = self.EKF_measurement_update(Gx, Gs, mean, covariance, measurement_current, measurement_predicted)
                X_map[j].mean = mean_updated
                X_map[j].covariance = covariance_updated

                #Particle Weighting
                weight = 1.0/(sqrt(2*pi*np.linalg.det(Q)))*exp((-0.5)*float((measurement_current-measurement_predicted).T.dot(inv(Q)).dot(measurement_current-measurement_predicted)))
                weight_total = weight_total * weight
        return weight_total

    def check_for_resampling(self, weight_sum):
        try:
            Meff = 1.0/weight_sum
        except ZeroDivisionError:
            return 1
        if Meff < (self.M/2):
            return 1

    def EKF_measurement_update(self, Gx, Gs, mean, covariance, measurement_current, measurement_predicted):
        P = np.identity(3) #TODO: change P here
        sigma = 0.1 #TODO:change sigma here
        Q = Gx.dot(P).dot(Gx.T) + Gx.dot(covariance).dot(Gx.T) + sigma*np.eye(2)
        K = covariance.dot(Gx.T).dot(np.linalg.det(Q))
        mean = mean + K.dot(measurement_current-measurement_predicted)
        covariance = (np.eye(3) - K.dot(Gx)).dot(covariance)
        return mean, covariance, Q


    def convert_feature_to_camera_frame(self, input_pose, mean):
        coords = input_pose.coordinates
        Rcm = self.form_matrix_Rcm(input_pose)
        X_c = Rcm.dot(mean) + coords
        return X_c

    def measurement_model(self, Xc):
        #TODO:debug for another features
        if (Xc.shape == (3,1)):
            measurement = (self.fc/float(Xc[2]))*(Xc[:2])
            return measurement

    def calc_jacobian_Gs(self, feature, pose):
        x = float(feature[0])
        y = float(feature[1])
        z = float(feature[2])
        #x = feature[0]
        #y = feature[1]
        #z = feature[2]
        px = pose.coordinates[0]
        py = pose.coordinates[1]
        pz = pose.coordinates[2]
        fc = self.fc
        dg_by_dx = np.array([[fc/z, 0, ((-1)*fc*x)/(z*z)],
                              [0, fc/(z*z), ((-1)*fc*y)/(z*z)]])
        dx_by_ds = np.array([[(-1)*y+px, z - pz, 0, 1, 0, 0],
                              [x - px, 0, -z +pz, 0, 1, 0],
                              [0, -x + px, y - py, 0, 0, 1]])
        Gs = dg_by_dx.dot(dx_by_ds)
        return Gs

    def calc_jacobian_Gx(self, feature):
        #x = float(feature[0])
        #y = float(feature[1])
        #z = float(feature[2])
        print "feature", feature
        x = float(feature[0])
        y = float(feature[1])
        z = float(feature[2])
        fc = self.fc
        dg_by_dx = np.array([[fc/z, 0, ((-1)*fc*x)/(z*z)],
                             [0, fc/(z*z), ((-1)*fc*y)/(z*z)]])
        Gx = dg_by_dx
        return Gx

    def form_matrix_Rcm(self, input_pose):
        euler_angles = input_pose.euler_angles
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



