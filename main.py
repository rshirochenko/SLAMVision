import initialization
import motion_model as m_m
import pose
import initialization as init
import fastSLAM

def main():
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

if  __name__ =='__main__':main()

