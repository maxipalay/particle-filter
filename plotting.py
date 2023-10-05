import matplotlib.pyplot as plt
import data_utils as du
import numpy as np

PLOT_REALTIME_PARTICLES, PLOT_REALTIME_MEAN, PLOT_FINAL_PATH = 0, 1, 2

class Plotter():
    def __init__(self, n_particles, landmarks, plot_mode=PLOT_REALTIME_PARTICLES):
        # configure plotting
        self.plot_mode = plot_mode
        
        if plot_mode == PLOT_REALTIME_MEAN or plot_mode == PLOT_REALTIME_PARTICLES:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8,8))
            
            self.line2, = self.ax.plot([], [], marker='o', markersize=8, linestyle = 'None')
            self.line, = self.ax.plot([], [], marker='o', markersize=2, linestyle = 'None')
            self.ax.plot(landmarks[:,du.LMT_X],landmarks[:,du.LMT_Y], marker='o', markersize=4, linestyle = 'None')
            self.visible_landmarks, = self.ax.plot([],[],marker='o', markersize=6, linestyle = 'None')
            self.arrow_gt = self.ax.arrow(0,0,0,0)
            if plot_mode == PLOT_REALTIME_MEAN:
                self.arrow_mean = self.ax.arrow(0,0,0,0)
            else:
                self.arrows = [self.ax.arrow(0,0,0,0) for _ in range(n_particles)]

    def update(self, means, state, z, tru, landmarks, i):
        # plot results
        if self.plot_mode == PLOT_REALTIME_MEAN:        
            #ax.plot(np.mean(np.matrix(state)[:,0]),np.mean(np.matrix(state)[:,1]), markersize=4, color='black', marker='o', linestyle='None')
            
            # get the mean x,y,theta (this is just for ease of understanding)
            x=means[i,0]
            y=means[i,1]
            theta=means[i,2]

            # set the x,y point of the mean
            self.line.set_data(means[i,0],means[i,1])

            # plot the direction
            dx = np.cos(theta)*0.5
            dy = np.sin(theta)*0.5
            self.arrow_mean.set_data(x=x,y=y, dx=dx, dy=dy)

        elif self.plot_mode == PLOT_REALTIME_PARTICLES:
            
            # plot all the points of our state
            self.line.set_data(state[:,0],state[:,1])

            # plot all the arrows
            for i,v in enumerate(self.arrows):
                x = state[i,0]
                y = state[i,1]
                theta = state[i,2]
                dx = np.cos(theta)*0.5
                dy = np.sin(theta)*0.5
                
                v.set_data(x=x,y=y, dx=dx, dy=dy)
            
        else:
            pass

        # if we're plotting in real time
        if self.plot_mode == PLOT_REALTIME_MEAN or self.plot_mode == PLOT_REALTIME_PARTICLES:
            # if there are measurements
            if z.size>0:
                # get the landmarks
                landmarks = np.matrix([landmarks[np.where(landmarks[:,du.LMT_S] == x[du.MEAS_S])[0][0]] for x in z])
                # plot them
                self.visible_landmarks.set_data(landmarks[:, du.LMT_X], landmarks[:, du.LMT_Y])
            # if there's ground truth data
            if tru.size > 0:
                x = tru[0,du.GT_X]
                y = tru[0,du.GT_Y]
                dx = np.cos(tru[0,du.GT_H])
                dy = np.sin(tru[0,du.GT_H])
                # set the point
                self.line2.set_data(x,y)
                # set the arrow
                self.arrow_gt.set_data(x=x, y=y, dx=dx, dy=dy)
                #ax.plot(tru[0,GT_X],tru[0,GT_Y], markersize=8, color='orange', marker='o', linestyle='None')
            
            plt.xlim(-6,9)
            plt.ylim(-7.5,7.5)
            
            plt.pause(1e-3)

        def plot(self, means, ground_truth):
            if self.plot_mode == PLOT_FINAL_PATH:
                plt.plot(means[:,0],means[:,1], 'b')
                plt.plot(ground_truth[:,du.GT_X],ground_truth[:,du.GT_Y], 'r')  # Adds to current figure

                plt.xlim((-10,10))
                plt.ylim((-10,10))
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.legend(['model','truth'])
                plt.show()