import initialization
import motion_model as m_m
import pose
import initialization as init
from measurement import *
import fastSLAM
import cv2
import time
import feature
import constants


path = '/home/rshir/data/simple_sift/result.avi'  # Video path
start_frame = 6
number_of_init_frames = 4  # Number of frames for initialization stage
finish_of_init_frame = start_frame + number_of_init_frames


# Open the video and read first fame
def open_video(path):
    cap = cv2.VideoCapture(path)
    # take first frame of the video
    if cap.isOpened(): # try to get the first frame
        ret, frame = cap.read()
    else:
        ret = False
    return cap, ret


def get_frame(cap, ret):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ret, gray
    else:
        return ret, None


def show_res(tab):
    for key in tab:
        print "key", key, ' point', tab[key].point, 'last_observed', tab[key].last_observed


def main():
    # Open the video
    cap, ret = open_video(path)
    ret, frame = get_frame(cap, ret)

    feature_dict = feature.FeaturesTemp()
    X_map_init = feature.XMapDictionary().X_map_dict
    particles_dict = []
    motion_model = m_m.Motion_model()
    fastslam = fastSLAM.FastSLAM()
    frame_number = 0

    while ret:
        frame_number += 1
        # Initialization stage
        if start_frame <= frame_number <= finish_of_init_frame:
            print "Initialization frame number:", frame_number
            if frame_number == start_frame:
                print "Initialization"
                K = []
                meas = Measurement(K)
                current_points = meas.init_measurement(frame)
                #initialization = init.Initialization()
                feature_dict.init_get_measurements(current_points, X_map_init)
            else:
                print "Working process"
                table_J, current_points = meas.make_measurement(frame)
                feature_dict.get_measurements(current_points, X_map_init)

        if frame_number == finish_of_init_frame:
            particles_dict = {}
            particle_weight = 1/constants.NUMBER_OF_PARTICLES
            init_pose = pose.Pose(np.random.rand(3), np.random.rand(3), np.random.rand(3))
            i = 0
            for i in range(1, constants.NUMBER_OF_PARTICLES):
                particles_dict[i] = pose.Particle(particle_weight, init_pose, X_map_init, X_map_init)

            #X_map = initialization.build_X_map(meas.table_K)

        # Algorithm working 'X_map' print "particles_dict", particles_dict
        if frame_number > finish_of_init_frame:
            # print "Algorithm working ",i
            table_J, current_points = meas.make_measurement(frame)
            i = 0
            for i in particles_dict:
                X_map_dict = particles_dict[i].X_map
                feature_dict.get_measurements(current_points, X_map_dict)
                for x in X_map_dict:
                    for y in X_map_dict:
                        try:
                            if np.linalg.norm(x.descriptor - y.descriptor) < 100:
                                print "I found ITTTT"
                                print "X_map_distance between", X_map_dict.index(x), " and ", X_map_dict.index(y), "\n"
                                print "X_map_distance points", x.debug_coord, " and ", y.debug_coord, "\n"

                        except:
                            pass

                # Motion model update
                motion_model.rotational_motion_model(particles_dict[i])
                motion_model.translational_optimization(particles_dict[i], current_points)

            """
            weight_sum = 0
            for particle_id in particles_dict:
                fastslam.measurement_update(particles_dict[particle_id], current_points)
                weight_sum = weight_sum + particles_dict[particle_id].weight


            # Check for resampling and resampling if condition true
            if fastslam.check_for_resampling(weight_sum):
                for particle_id in particles_dict:
                    particles_dict[particle_id].weight = 1.0/fastslam.M
            """
        ret, frame = get_frame(cap, ret)
    else:
        show_res(meas.table_K)
        print "time", time.clock()
        print("Video`s end or camera is not working ")


if  __name__ =='__main__' :main()

