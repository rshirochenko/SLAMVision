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
import copy


path = '/home/rshir/data/simple_sift/result.avi'  # Video path
start_frame = 6
number_of_init_frames = 4  # Number of frames for initialization stage
finish_of_init_frame = start_frame + number_of_init_frames


# Open the video and read first fame
def open_video(path):
    cap = cv2.VideoCapture(path)
    # take first frame of the video
    if cap.isOpened():  # try to get the first frame
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

    feature_cache = feature.FeaturesTemp()
    feature_cache_particle = {}
    X_map_init = feature.XMapDictionary().X_map_dict
    particles_dict = []
    motion_model = m_m.Motion_model()
    fastslam = fastSLAM.FastSLAM()
    init_pose = pose.Pose(constants.init_euler_angles,
                          constants.init_angular_rates,
                          constants.init_coordinates)
    frame_number = 0

    while ret:
        frame_number += 1
        print frame_number
        # Initialization stage
        if start_frame <= frame_number <= finish_of_init_frame:
            print "Initialization frame number:", frame_number
            if frame_number == start_frame:
                print "Initialization"
                K = []
                meas = Measurement(K)
                current_points = meas.init_measurement(frame)
                feature_cache.init_get_measurements(current_points, X_map_init, pose)
            else:
                print "Working process"
                table_J, current_points = meas.make_measurement(frame)
                feature_cache.get_measurements(current_points, X_map_init, init_pose)

        if frame_number == finish_of_init_frame:
            particles_dict = {}
            particle_weight = 1 / float(constants.NUMBER_OF_PARTICLES)
            print "number_of_particles", constants.NUMBER_OF_PARTICLES
            for i in range(0, constants.NUMBER_OF_PARTICLES):
                particles_dict[i] = pose.Particle(particle_weight, init_pose, copy.deepcopy(X_map_init),
                                                  copy.deepcopy(X_map_init))
                feature_cache_particle[i] = feature.FeaturesTemp()
                print "measurument_dict", feature_cache.measurement_dict
                print "times_observed", feature_cache.times_observed
                feature_cache_particle[i].measurement_dict = copy.deepcopy(feature_cache.measurement_dict)
                feature_cache_particle[i].times_observed = copy.deepcopy(feature_cache.times_observed)

        # Algorithm working 'X_map' print "particles_dict", particles_dict
        if frame_number > finish_of_init_frame:
            # print "Algorithm working ",i
            table_J, current_points = meas.make_measurement(frame)
            for i in particles_dict:
                feature_cache_particle[i].get_measurements(current_points, particles_dict[i].X_map_dict, particles_dict[i].pose)
                """
                for x in X_map_dict:
                    for y in X_map_dict:
                        try:
                            if np.linalg.norm(x.descriptor - y.descriptor) < 100 and x != y:
                                print "frame_number", frame_number
                                print "I found ITTTT"
                                print "X_map_distance between", X_map_dict.index(x), " and ", X_map_dict.index(y), "\n"
                                print "X_map_distance points", x.debug_coord, " and ", y.debug_coord, "\n"
                        except:
                            pass
                """
                # Motion model update
                motion_model.rotational_motion_model(particles_dict[i].pose)
                motion_model.translational_optimization(particles_dict[i], current_points)

                # ***Rao-Blackwellized particle filter update***
                weight_sum = 0
                for particle_id in particles_dict:
                    fastslam.calc_position_proposal_distribution(particles_dict[particle_id], current_points)
                    weight = fastslam.measurement_update(particles_dict[particle_id], current_points)
                    weight_sum = weight_sum + weight

                # Check for resampling and resampling if condition true
                if fastslam.check_for_resampling(weight_sum):
                    for particle_id in particles_dict:
                        print "Here"
                        particles_dict[particle_id].weight = 1.0/fastslam.M

        ret, frame = get_frame(cap, ret)
    else:
        # show_res(meas.table_K)
        print "time", time.clock()
        print("Video`s end or camera is not working ")


if __name__ == '__main__': main()

