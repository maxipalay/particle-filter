import numpy as np
import matplotlib.pyplot as plt

from particle_filter import ParticleFilter
from data_utils import DataLoader
from prob_utils import sample, low_variance_sampler
from measurement_model import measurement_likelihood
from motion_model import motion_model

PLOT_REALTIME_PARTICLES, PLOT_REALTIME_MEAN, PLOT_FINAL_PATH = 0, 1, 2

CONFIG_LENGTH = 3

CONTROL_V, CONTROL_W = 0, 1 # control is represented by an array of [velocity, angular velocity] commands in [m/s,rad/s]
STATE_X, STATE_Y, STATE_THETA = 0, 1, 2 # state is represented by an array if size [n particles, 3]

# USER-SET PARAMS
plot_mode = PLOT_REALTIME_PARTICLES
N_PARTICLES = 50
seed_initial_position = True
SIGMA_STATE = 1

if __name__=="__main__":

    dl = DataLoader()

    previous_timestamp = None
        # get the robot's starting position in world coordinates
    x0, yo, h0 = dl.ground_truth[0,dl.GT_X], dl.ground_truth[0,dl.GT_Y], dl.ground_truth[0,dl.GT_H]

    # initialize random state
    if seed_initial_position:
        # generate a state with a Gaussian distibution around the initial point
        state = [[x0+sample(SIGMA_STATE), y0+sample(SIGMA_STATE), h0+sample(SIGMA_STATE)] for _ in range(N_PARTICLES)]
    else:
        state = [[sample(SIGMA_STATE), sample(SIGMA_STATE), sample(SIGMA_STATE)] for _ in range(N_PARTICLES)]

    # instance the particle filter    
    filter = ParticleFilter(motion_model=motion_model, measurement_likelihood=measurement_likelihood, n_particles = N_PARTICLES, initial_state=state)

    # configure plotting
    if plot_mode == PLOT_REALTIME_MEAN or plot_mode == PLOT_REALTIME_PARTICLES:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,8))
        line, = ax.plot([], [], 'o-', markersize=2, linestyle = 'None')
        line2, = ax.plot([], [], 'o-', markersize=8, linestyle = 'None')
        ax.plot(dl.landmarks[:,dl.LMT_X],dl.landmarks[:,dl.LMT_Y], 'o-', markersize=4, linestyle = 'None')
        visible_landmarks, = ax.plot([],[],'o-', markersize=6, linestyle = 'None')
        arrow_gt = ax.arrow(0,0,0,0)
        if plot_mode == PLOT_REALTIME_MEAN:
            arrow_mean = ax.arrow(0,0,0,0)
        else:
            arrows = [ax.arrow(0,0,0,0) for _ in range(N_PARTICLES)]

    positions = []
    means = []
    odom = dl.odometry
    positions = np.zeros((len(odom),N_PARTICLES,CONFIG_LENGTH))
    means = np.zeros((len(odom),3))

    # run over the odometry data
    for i,row in enumerate(odom):
        # first time check
        if previous_timestamp is None:
            previous_timestamp = row[dl.ODOM_T]
        else:
            # get time in between timesteps
            dt = row[dl.ODOM_T] - previous_timestamp
            # get control action
            u = [row[dl.ODOM_V], row[dl.ODOM_W]]
            # get measurements that happened between this and the previous timestep
            z = dl.get_measurements(previous_timestamp, row[dl.ODOM_T])
            # get the ground truth measurements between our previous loop and this one
            tru = dl.get_groundtruth(previous_timestamp, row[dl.ODOM_T])

            # call our particle filter to get new state
            state = filter.update(u, z, dt, dl.landmarks)
            
            # add the means to our mean-tracking matrix
            means[i,:] = np.array([np.mean(state[:,0]),np.mean(state[:,1]), np.mean(state[:,2])])
            
            # plot results
            if plot_mode == PLOT_REALTIME_MEAN:        
                #ax.plot(np.mean(np.matrix(state)[:,0]),np.mean(np.matrix(state)[:,1]), markersize=4, color='black', marker='o', linestyle='None')
                
                # get the mean x,y,theta (this is just for ease of understanding)
                x=means[i,0]
                y=means[i,1]
                theta=means[i,2]

                # set the x,y point of the mean
                line.set_data(means[i,0],means[i,1])

                # plot the direction
                dx = np.cos(theta)*0.5
                dy = np.sin(theta)*0.5
                arrow_mean.set_data(x=x,y=y, dx=dx, dy=dy)

            elif plot_mode == PLOT_REALTIME_PARTICLES:
                
                # plot all the points of our state
                line.set_data(state[:,0],state[:,1])

                # plot all the arrows
                for i,v in enumerate(arrows):
                    x = state[i,0]
                    y = state[i,1]
                    theta = state[i,2]
                    dx = np.cos(theta)*0.5
                    dy = np.sin(theta)*0.5
                    
                    v.set_data(x=x,y=y, dx=dx, dy=dy)
                
            else:
                pass

            # if we're plotting in real time
            if plot_mode == PLOT_REALTIME_MEAN or plot_mode == PLOT_REALTIME_PARTICLES:
                # if there are measurements
                if z.size>0:
                    # get the landmarks
                    landmarks = np.matrix([dl.landmarks[np.where(dl.landmarks[:,dl.LMT_S] == x[dl.MEAS_S])[0][0]] for x in z])
                    # plot them
                    visible_landmarks.set_data(landmarks[:, dl.LMT_X], landmarks[:, dl.LMT_Y])
                # if there's ground truth data
                if tru.size > 0:
                    x = tru[0,dl.GT_X]
                    y = tru[0,dl.GT_Y]
                    dx = np.cos(tru[0,dl.GT_H])
                    dy = np.sin(tru[0,dl.GT_H])
                    # set the point
                    line2.set_data(x,y)
                    # set the arrow
                    arrow_gt.set_data(x=x, y=y, dx=dx, dy=dy)
                    #ax.plot(tru[0,GT_X],tru[0,GT_Y], markersize=8, color='orange', marker='o', linestyle='None')
                plt.xlim(-6,9)
                plt.ylim(-7.5,7.5)
                plt.pause(1e-3)

            # update timestamp
            previous_timestamp = row[dl.ODOM_T]

    if plot_mode == PLOT_FINAL_PATH:
        plt.plot(means[:,0],means[:,1], 'b')
        plt.plot(dl.ground_truth[:,dl.GT_X],dl.ground_truth[:,dl.GT_Y], 'r')  # Adds to current figure

        plt.xlim((-10,10))
        plt.ylim((-10,10))
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend(['model','truth'])
        plt.show()