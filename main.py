import initialization
import motion_model as m_m
import pose
import initialization as init
from measurement import *
import fastSLAM
import cv2
import time


path = '/home/rshir/data/simple_sift/result.avi'  # Video path
start_frame = 4
number_of_init_frames = 7  # Number of frames for initialization stage
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
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
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

    fastslam = fastSLAM.FastSLAM()
    motion_model = m_m.Motion_model()
    X_map = []

    i = 0
    while ret:
        i += 1
        # Initialization stage

        if start_frame < i <= finish_of_init_frame:
            print "Initialization frame number:", i
            if i == 5:
                print "Initialization"
                K = []
                meas = Measurement(K)
                meas.init_measurement(frame)
                initialization = init.Initialization()
            else:
                print "Working process"
                table_J, current_points = meas.make_measurement(frame)
                initialization.get_img_coordinates(table_J)

        if i == finish_of_init_frame:
            X_map = initialization.build_X_map()

        # Algorithm working 'X_map'
        if i > number_of_init_frames:
            # print "Algorithm working ",i
            table_J, current_points = meas.make_measurement(frame)
            particles_dict = pose.make_particles_dict(X_map)
            #print "particles_dict", particles_dict
            #print "X_map", type(X_map)
            #print "Z measurements", current_points

            for point in X_map:
                print "X_map", point
            for current_point in current_points:
                print "current_points", current_point.point



            # Motion model update
            particles_dict = motion_model.rotational_motion_model(particles_dict)
            motion_model.translational_optimization(particles_dict, current_points)
            """
            # Measurement model update(SLAM part)
            weight_sum = 0
            for particle_id in particles_dict:
                print fastslam.measurement_update(particles_dict[particle_id], meas.table_K)
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

    """
    #Initilization stage:
    initialize = init.Initialization()
    a, Z_table_K = initialize.get_img_coordinates_Z()
    fastslam = fastSLAM.FastSLAM()
    motion_model = m_m.Motion_model()
    particles_dict = pose.makeParticlesDict(10)


    #Motion model update
    motion_model.rotational_motion_model(particles_dict)
    motion_model.translational_optimization(particles_dict,Z_table_K)

    #Mesurment model update(SLAM part)
    weight_sum = 0
    for particle_id in particles_dict:
        print fastslam.measurement_update(particles_dict[particle_id], Z_table_K)
        weight_sum = weight_sum + particles_dict[particle_id].weight

    #Check for resampling and resampling if condition true
    if fastslam.check_for_resampling(weight_sum):
        for particle_id in particles_dict:
            particles_dict[particle_id].weight = 1.0/fastslam.M
    '"""

if  __name__ =='__main__' :main()

