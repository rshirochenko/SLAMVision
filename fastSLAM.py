from motion_model import *
import pose
from math import cos, sin, sqrt, exp, pi
import constants


class FastSLAM(object):
    fc = constants.fc
    Rprop = []
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
                try:
                    covariance_sum += Gs.T.dot(inv(Q)).dot(Gs) + inv(constants.P)
                    measurement_current = np.asarray(current_measurements[x.debug_key].point).reshape(2, 1)
                    mean_part += Gs.T.dot(inv(Q)).dot(measurement_current-measurement_predicted)
                except:
                    return 1
        try:
            covariance_sum = inv(covariance_sum)
            st = np.vstack((particle.pose.euler_angles, particle.pose.coordinates))
            mean_st = covariance_sum.dot(mean_part) + st
            particle.pose.euler_angles = mean_st[0:3]
            particle.pose.coordinates = mean_st[3:6]
        except np.linalg.linalg.LinAlgError:
            pass


    def calc_position_proposal_distribution_fastslam1(self, particle, current_measurements):
        particle.pose.coordinates = particle.pose.coordinates + constants.vnoise_process

    """ Measurement update and particle weighting stage"""
    def measurement_update(self, particle, current_measurements):
        X_map = particle.X_map_dict
        weight_total = 1
        for x in X_map:
            if x.debug_key in current_measurements:
                print "I am here"
                mean = x.mean
                covariance = x.covariance
                X_c = self.convert_feature_to_camera_frame(particle.pose, mean)
                measurement_predicted = self.measurement_model(X_c)
                measurement_current = np.asarray(current_measurements[x.debug_key].point).reshape(2, 1)
                Gx = self.calc_jacobian_Gx(X_c)
                Gs = self.calc_jacobian_Gs(X_c, particle.pose)

                #EKF measurement update
                mean_updated, covariance_updated, Q = self.EKF_measurement_update(Gx, Gs, mean, covariance, measurement_current, measurement_predicted)
                x.mean = mean_updated
                x.covariance = covariance_updated

                #Particle Weighting
                try:
                    weight = 1.0/(sqrt(2*pi*np.linalg.det(Q)))*exp((-0.5)*float((measurement_current-measurement_predicted).T.dot(inv(Q)).dot(measurement_current-measurement_predicted)))
                except ValueError:
                    weight = 1/constants.NUMBER_OF_PARTICLES
                weight_total = weight_total * weight
        return weight_total

    """ Particle weight resampling stage"""
    def check_for_resampling(self, weight_sum):
        try:
            Meff = 1.0/weight_sum
        except ZeroDivisionError:
            return 1
        if Meff < (self.M/2):
            return 1

    def EKF_measurement_update(self, Gx, Gs, mean, covariance, measurement_current, measurement_predicted):
        sigma = constants.sigma_meas
        P = constants.P
        Q = Gs.dot(P).dot(Gs.T) + Gx.dot(covariance).dot(Gx.T) + sigma*np.eye(2)
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
        px = float(pose.coordinates[0])
        py = float(pose.coordinates[1])
        pz = float(pose.coordinates[2])
        fc = self.fc
        dg_by_dx = np.array([[fc/z, 0, ((-1)*fc*x)/(z*z)],
                              [0, fc/(z*z), ((-1)*fc*y)/(z*z)]])
        dx_by_ds = np.array([[(-1)*y+px, z - pz, 0, 1, 0, 0],
                              [x - px, 0, -z +pz, 0, 1, 0],
                              [0, -x + px, y - py, 0, 0, 1]])
        Gs = dg_by_dx.dot(dx_by_ds)
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



