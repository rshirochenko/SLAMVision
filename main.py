import motion_model as m_m
import pose
from measurements_z import *
import fastSLAM
import cv2
import time
import feature
import constants
import copy


path = '/home/rshir/data/simple_sift/result.avi'  # Video path
start_frame = 6  # Passing by first 5 frames, because of OpenCV SIFT estimation method work.
number_of_init_frames = 4  # Number of frames for initialization stage
finish_of_init_frame = start_frame + number_of_init_frames


def open_video(path):
    """
    Open the video and read first fame.
    """
    cap = cv2.VideoCapture(path)  # take first frame of the video
    if cap.isOpened():  # try to get the first frame
        ret, frame = cap.read()
    else:
        ret = False
    return cap, ret


def get_frame(cap, ret):
    """
    Getting frame and converting the frame(image) to gray color image.
    """
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ret, gray
    else:
        return ret, None


def show_res(tab):
    """
    Method for debugging. It will be deleted later.
    """
    for key in tab:
        print "key", key, ' point', tab[key].point, 'last_observed', tab[key].last_observed


def main():
    cap, ret = open_video(path)  # Open the video file
    ret, frame = get_frame(cap, ret)  # Get the frame in gray color

    feature_cache = feature.FeaturesCache()  # Features cache for keeping 4 features for initialization
    feature_cache_particle = {}  # Features cache variable for particles filter
    X_map_init = feature.XMapDictionary().X_map_dict  # Features map (X) on the initialization stage
    particles_filter = []  # Particles filter variable
    motion_model = m_m.Motion_model()  # Motion model object
    fastslam = fastSLAM.FastSLAM()  # SLAM object
    init_pose = pose.Pose(constants.init_euler_angles,  # Initial pose value from constants.py file
                          constants.init_angular_rates,
                          constants.init_coordinates)
    frame_number = 0

    # For each frame running the algorithm cycle
    while ret:
        """Initialization stage. It lasts for the 4 frames, in order to create measurements map object(X)."""
        frame_number += 1
        print frame_number
        # Passing by first 5 frames, because of OpenCV SIFT estimation method work.
        if start_frame <= frame_number <= finish_of_init_frame:
            print "Initialization frame number:", frame_number
            if frame_number == start_frame:  # Initialization frame for creating all variable
                measurements_Z = Measurements_Z([])  # Initialize measurements Z object
                current_measurements = measurements_Z.init_measurement(frame)  # Estimate and input SIFT measurements
                feature_cache.init_get_measurements(current_measurements, X_map_init, pose)  # Input in the feature cache
            else:  # Working until form first measurements map(X) object elements
                current_measurements = measurements_Z.make_measurement(frame)
                feature_cache.get_measurements(current_measurements, X_map_init, init_pose)

        # Creating the particles objects for the particle filter
        if frame_number == finish_of_init_frame:
            particles_filter = {}
            particle_weight = 1 / float(constants.NUMBER_OF_PARTICLES)
            for i in range(0, constants.NUMBER_OF_PARTICLES):
                particles_filter[i] = pose.Particle(particle_weight, copy.deepcopy(init_pose), copy.deepcopy(X_map_init)
                                                    , copy.deepcopy(X_map_init))
                feature_cache_particle[i] = copy.deepcopy(feature.FeaturesCache())
                feature_cache_particle[i].measurement_dict = copy.deepcopy(feature_cache.measurement_dict)
                feature_cache_particle[i].times_observed = copy.deepcopy(feature_cache.times_observed)

        """ Algorithm work main stage. """
        if frame_number > finish_of_init_frame:
            current_measurements = measurements_Z.make_measurement(frame)
            for i in particles_filter:
                feature_cache_particle[i].get_measurements(current_measurements, particles_filter[i].X_map_dict, particles_filter[i].pose)

                # Motion model update
                motion_model.rotational_motion_model(particles_filter[i].pose)
                motion_model.translational_optimization(particles_filter[i], current_measurements)

                # ***Rao-Blackwellized particle filter update***
                # Time update stage
                fastslam.calc_position_proposal_distribution(particles_filter[i], current_measurements)
                # Measurement update stage and particles weighting
                weight_sum = 0
                weight = fastslam.measurement_update(particles_filter[i], current_measurements)
                weight_sum = weight_sum + weight

                # Check for resampling and resampling if condition true
                if fastslam.check_for_resampling(weight_sum):
                    particles_filter[i].weight = 1.0/constants.NUMBER_OF_PARTICLES
        ret, frame = get_frame(cap, ret)
    else:
        print "time", time.clock()
        print("Video`s end or camera is not working ")


if __name__ == '__main__': main()

