import pose
import numpy as np
import measurement
import initialization
from numpy.linalg import inv

class Motion_model(object):
    fc = 0.5

    def __init__(self):
        self.dt = 1.0/30.0
        print("Motion model is initiated")

    """Updates the angels and rates in the particles by using the motion model '"""
    def rotational_motion_model(self, particles):
        M = np.identity(3) # TODO: here should be a matrix function (seem eq.4.13)
        vnoise = np.random.rand(3) #TODO: here should be zero meenas, gaussian random forcing terms (check eq. 4.15)
        for particle_id in particles:
            #Get the previous step pose angles and rates
            pose = particles[particle_id].pose
            euler_angles_prev_step = pose.euler_angles # vector of euler_angles [psi theta phi]
            angular_rates_prev_step = pose.angular_rates

            # Calculate the current step angles and rates
            euler_angles = euler_angles_prev_step + np.dot(M,angular_rates_prev_step)*self.dt
            angular_rates = angular_rates_prev_step + vnoise

            #update particles dictionary
            particles[particle_id].pose.euler_angles = euler_angles
            particles[particle_id].pose.angular_rates = angular_rates

    def translational_optimization(self, particles, Z_table_K):
        fc = self.fc
        for particle_id in particles:
            #TODO: add eq. 5.4 here, as now it calculates in map frame coordinates. Need to calculate in camera frame coordinates
            particle_X_map = particles[particle_id].X_map
            i = 0
            for feature_id in particle_X_map:
                if feature_id in Z_table_K:
                    x = np.asarray(particle_X_map[feature_id].mean) #feature x[x y z]
                    img_coord = np.asarray(Z_table_K[feature_id].point) # image coordinate [u v]

                    b11 = (fc*x[0]-img_coord[0]*x[2])
                    b21 = (fc*x[0]-img_coord[1]*x[2])
                    bx = np.array([b11,
                                   b21])

                    Ax = np.array([[(-1)*fc, 0, img_coord[0]],
                                   [0, (-1)*fc, img_coord[1]]])
                    if i == 0:
                        A = Ax
                        b = bx
                    else:
                        A = np.vstack((A,Ax))
                        b = np.vstack((b,bx))
                    i = i + 1
            delta_p = inv(A.T.dot(A)).dot(A.T).dot(b)
            #update particles dictionary
            particles[particle_id].pose.position = delta_p


def test_motion_model():
    motion_model = Motion_model()

def main():
    initialize = initialization.Initialization()
    a, Z_table_K = initialize.get_img_coordinates_Z()

    particles_dict = pose.makeParticlesDict(2)

    motion_model = Motion_model()
    motion_model.rotational_motion_model(particles_dict)
    motion_model.translational_optimization(particles_dict,Z_table_K)

if  __name__ =='__main__':main()
