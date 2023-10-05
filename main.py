import numpy as np
import matplotlib.pyplot as plt

from particle_filter import ParticleFilter
from prob_utils import sample, low_variance_sampler
from measurement_model import measurement_likelihood
from motion_model import motion_model
import data_utils as du
import plotting


CONFIG_LENGTH = 3
# control is represented by an array of [velocity, angular velocity] commands in [m/s,rad/s]
CONTROL_V, CONTROL_W = 0, 1
# state is represented by an array if size [n particles, 3]
STATE_X, STATE_Y, STATE_THETA = 0, 1, 2

# USER-SET PARAMS
N_PARTICLES = 200
seed_initial_position = True
SIGMA_INITIAL_STATE = 1


def initialize_random_state(x0=0, y0=0, h0=0):
    # generate a state with a Gaussian distibution around the initial point
    state = [[x0+sample(SIGMA_INITIAL_STATE), y0+sample(SIGMA_INITIAL_STATE),
              h0+sample(SIGMA_INITIAL_STATE)] for _ in range(N_PARTICLES)]
    return state


if __name__ == "__main__":

    dl = du.DataLoader()

    # get the robot's starting position in world coordinates
    x0, y0, h0 = dl.ground_truth[0, du.GT_X], dl.ground_truth[0,
                                                              du.GT_Y], dl.ground_truth[0, du.GT_H]

    state = initialize_random_state(x0, y0, h0)

    # instance the particle filter
    filter = ParticleFilter(motion_model=motion_model, measurement_likelihood=measurement_likelihood,
                            n_particles=N_PARTICLES, initial_state=state)

    # instance plotter
    plot = plotting.Plotter(N_PARTICLES, dl.landmarks,
                            plotting.PLOT_REALTIME_PARTICLES)

    odom = dl.odometry
    positions = np.zeros((len(odom), N_PARTICLES, CONFIG_LENGTH))
    means = np.zeros((len(odom), 3))
    previous_timestamp = None

    # run over the odometry data
    for i, row in enumerate(odom):
        # first time check
        if previous_timestamp is None:
            previous_timestamp = row[du.ODOM_T]
        else:
            # get time in between timesteps
            dt = row[du.ODOM_T] - previous_timestamp
            # get control action
            u = [row[du.ODOM_V], row[du.ODOM_W]]
            # get measurements that happened between this and the previous timestep
            z = dl.get_measurements(previous_timestamp, row[du.ODOM_T])
            # get the ground truth measurements between our previous loop and this one
            tru = dl.get_groundtruth(previous_timestamp, row[du.ODOM_T])

            # call our particle filter to get new state
            state = filter.update(u, z, dt, dl.landmarks)

            # add the means to our mean-tracking matrix
            means[i, :] = np.array(
                [np.mean(state[:, 0]), np.mean(state[:, 1]), np.mean(state[:, 2])])
            # update the plot (if configured for real time plotting)
            plot.update(means, state, z, tru, dl.landmarks, i)
            # update timestamp
            previous_timestamp = row[du.ODOM_T]

    plot.plot()
