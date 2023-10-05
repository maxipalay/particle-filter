from particle_filter import ParticleFilter
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from measurement_model import landmark_likelihood
from motion_model import motion_model
LMT_S, LMT_X, LMT_Y = 0, 1, 2
GT_T, GT_X, GT_Y, GT_H = 0, 1, 2, 3
# indexes in files
ODOM_T, ODOM_V, ODOM_W = 0, 1, 2
MEAS_T, MEAS_S, MEAS_R, MEAS_B = 0, 1, 2, 3

BARC_S, BARC_B = 0, 1

N_PARTICLES = 10   # amount of particles we're using
CONFIG_LENGTH = 3  # amount of variables that represent our state

from data_utils import *

# control is represented by an array of [velocity, angular velocity] commands in [m/s,rad/s]
# state is represented by an array if size [n particles, 3]
CONTROL_V, CONTROL_W = 0, 1
STATE_X, STATE_Y, STATE_THETA = 0, 1, 2




filter = ParticleFilter(motion_model=motion_model, landmark_likelihood=landmark_likelihood,
                        n_particles=N_PARTICLES, config_length=CONFIG_LENGTH)

realtime_plot = False

previous_timestamp = None
    # get the robot's starting position in world coordinates
x0 = ground_truth[0,GT_X]
y0 = ground_truth[0,GT_Y]
h0 = ground_truth[0,GT_H]

# initialize random state
state = [[x0+sample(1), y0+sample(1), h0+sample(1)] for _ in range(N_PARTICLES)]

if realtime_plot:
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,8))
    line, = ax.plot([], [], 'o-', markersize=2, linestyle = 'None')
    line2, = ax.plot([], [], 'o-', markersize=8, linestyle = 'None')
    ax.plot(landmarks[:,LMT_X],landmarks[:,LMT_Y], 'o-', markersize=4, linestyle = 'None')
    visible_landmarks, = ax.plot([],[],'o-', markersize=6, linestyle = 'None')
    arrow_gt = ax.arrow(0,0,0,0)
    arrows = [ax.arrow(0,0,0,0) for _ in range(N_PARTICLES)]
    arrow_mean = ax.arrow(0,0,0,0)

positions = []
means = []
odom = odometry
positions = np.zeros((len(odom),N_PARTICLES,3))
means = np.zeros((len(odom),2))

for i,row in enumerate(odom):
    if previous_timestamp is None:
        previous_timestamp = row[ODOM_T]
    else:
        # get tim ebetween timesteps
        dt = row[ODOM_T] - previous_timestamp
        # get control action
        u = [row[ODOM_V], row[ODOM_W]]
        # get measurements that happened between our previous loop and this one
        z = measure(previous_timestamp, row[ODOM_T])
        # get the ground measured between our previous loop and this one
        tru = groundtruth(previous_timestamp, row[ODOM_T])

        # call our particle filter to get new state
        #if u[0] != 0.0 or u[1] != 0.0 or z.size > 0:
        state = filter.update(u, z, dt)
        
        # add new state to out positions array
        #positions.append(state.copy())
        #positions[i,:,:] = state.copy()
        # add mean of the state to means array
        #means.append([np.mean(np.matrix(state)[:,0]),np.mean(np.matrix(state)[:,1])])
        a = np.matrix(state)
        means[i,:] = np.array([np.mean(a[:,0]),np.mean(a[:,1])])
        # plot results
        if realtime_plot:
            plt.xlim(-6,9)

            plt.ylim(-7.5,7.5)
        
        #ax.plot(np.mean(np.matrix(state)[:,0]),np.mean(np.matrix(state)[:,1]), markersize=4, color='black', marker='o', linestyle='None')
        
        #line.set_data(state[:,0],state[:,1])
        #for i,v in enumerate(arrows):
        #    v.set_data(x=state[i,0],y=state[i,1], dx=np.cos(state[i,2])*0.5, dy=np.sin(state[i,2])*0.5)
                
            line.set_data(np.mean(np.matrix(state)[:,0]),np.mean(np.matrix(state)[:,1]))
            
            x=np.mean(np.matrix(state)[:,0])
            y=np.mean(np.matrix(state)[:,1])
            theta=np.mean(np.matrix(state)[:,2])
            dx = np.cos(theta)*0.5
            dy = np.sin(theta)*0.5
            arrow_mean.set_data(x=x,y=y, dx=dx, dy=dy)
            #print(theta/np.pi*180.0)
                    
            if z.size>0:
                landmarks = np.matrix([get_landmark_from_measurement(x) for x in z])
                visible_landmarks.set_data(landmarks[:, LMT_X], landmarks[:, LMT_Y])
            if tru.size>0:
                line2.set_data(tru[0,GT_X],tru[0,GT_Y])
                arrow_gt.set_data(x=tru[0,GT_X], y=tru[0,GT_Y], dx=np.cos(tru[0,GT_H]), dy=np.sin(tru[0,GT_H]))
                #ax.plot(tru[0,GT_X],tru[0,GT_Y], markersize=8, color='orange', marker='o', linestyle='None')
            
                    
            plt.pause(1e-3)

        # update timestamp
        previous_timestamp = row[ODOM_T]

#m = np.matrix(means)
plt.plot(means[:,0],means[:,1], 'b')  # Adds to current figure
#plt.plot(averaged_m[:,0],averaged_m[:,1], 'b')  # Adds to current figure

#m2 = np.matrix(truth)
plt.plot(ground_truth[:,GT_X],ground_truth[:,GT_Y], 'r')  # Adds to current figure

plt.xlim((-10,10))
plt.ylim((-10,10))
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['model','truth'])
plt.show()