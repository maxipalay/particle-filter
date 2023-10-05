import numpy as np
import matplotlib.pyplot as plt
from data_utils import GroundTruth, Landmark, Measurement
from state import State

class Plotter():
    PLOT_REALTIME_PARTICLES, PLOT_REALTIME_MEAN, PLOT_FINAL_PATH = 0, 1, 2

    def __init__(self, n_particles, landmarks, plot_mode=PLOT_REALTIME_PARTICLES):
        # configure plotting
        self.plot_mode = plot_mode

        if plot_mode == self.PLOT_REALTIME_MEAN or plot_mode == self.PLOT_REALTIME_PARTICLES:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

            # landmarks
            self.ax.plot(landmarks[:, Landmark.X], landmarks[:, Landmark.Y],
                         marker='o', markersize=4, linestyle='None', color='#0047AB')

            # visible landmarks (seen by the robot)
            self.visible_landmarks, = self.ax.plot(
                [], [], marker='o', markersize=6, linestyle='None', color='#FF0500')

            if plot_mode == self.PLOT_REALTIME_MEAN:
                # ground truth
                self.truth_dot, = self.ax.plot(
                    [], [], marker='o', markersize=6, linestyle='None', color='#228B22')
                self.truth_arrow = self.ax.arrow(0, 0, 0, 0, color='#228B22')
                # particles
                self.particles_dots, = self.ax.plot(
                    [], [], marker='o', markersize=2, linestyle='None', color='#FF5F15')
                self.mean_arrow = self.ax.arrow(0, 0, 0, 0, color='#222222')

            else:
                # particles
                self.particles_dots, = self.ax.plot(
                    [], [], marker='o', markersize=2, linestyle='None', color='#FF5F15')
                self.particles_arrows = [self.ax.arrow(0, 0, 0, 0, color='#222222')
                                         for _ in range(n_particles)]

                # ground truth
                self.truth_dot, = self.ax.plot(
                    [], [], marker='o', markersize=6, linestyle='None', color='#228B22')
                self.truth_arrow = self.ax.arrow(0, 0, 0, 0, color='#228B22')

    def update(self, means, state, z, tru, landmarks, i):
        # plot results
        if self.plot_mode == self.PLOT_REALTIME_MEAN:
            # ax.plot(np.mean(np.matrix(state)[:,0]),np.mean(np.matrix(state)[:,1]), markersize=4, color='black', marker='o', linestyle='None')

            # get the mean x,y,theta (this is just for ease of understanding)
            x = means[i, State.X]
            y = means[i, State.Y]
            theta = means[i, State.HEADING]

            # set the x,y point of the mean
            self.particles_dots.set_data(means[i, State.X], means[i, State.Y])

            # plot the direction
            dx = np.cos(theta)*0.5
            dy = np.sin(theta)*0.5
            self.mean_arrow.set_data(x=x, y=y, dx=dx, dy=dy)

        elif self.plot_mode == self.PLOT_REALTIME_PARTICLES:

            # plot all the points of our state
            self.particles_dots.set_data(state[:, State.X], state[:, State.Y])

            # plot all the arrows
            for i, v in enumerate(self.particles_arrows):
                x = state[i, State.X]
                y = state[i, State.Y]
                theta = state[i, State.HEADING]
                dx = np.cos(theta)*0.5
                dy = np.sin(theta)*0.5

                v.set_data(x=x, y=y, dx=dx, dy=dy)

        else:
            pass

        # if we're plotting in real time
        if self.plot_mode == self.PLOT_REALTIME_MEAN or self.plot_mode == self.PLOT_REALTIME_PARTICLES:
            # if there are measurements
            if z.size > 0:
                # get the landmarks
                landmarks = np.matrix(
                    [landmarks[np.where(landmarks[:, Landmark.SUBJECT] == x[Measurement.SUBJECT])[0][0]] for x in z])
                # plot them
                self.visible_landmarks.set_data(
                    landmarks[:, Landmark.X], landmarks[:, Landmark.Y])
            # if there's ground truth data
            if tru.size > 0:
                x = tru[0, GroundTruth.X]
                y = tru[0, GroundTruth.Y]
                dx = np.cos(tru[0, GroundTruth.H])
                dy = np.sin(tru[0, GroundTruth.H])
                # set the point
                self.truth_dot.set_data(x, y)
                # set the arrow
                self.truth_arrow.set_data(x=x, y=y, dx=dx, dy=dy)
                # ax.plot(tru[0,GT_X],tru[0,GT_Y], markersize=8, color='orange', marker='o', linestyle='None')

            plt.xlim(-6, 9)
            plt.ylim(-7.5, 7.5)

            plt.pause(1e-3)

    def plot(self, means, ground_truth):
        if self.plot_mode == self.PLOT_FINAL_PATH:
            plt.plot(means[:, State.X], means[:, State.Y], 'b')
            # Adds to current figure
            plt.plot(ground_truth[:, GroundTruth.X],
                        ground_truth[:, GroundTruth.Y], 'r')

            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.legend(['model', 'truth'])
            plt.show()
