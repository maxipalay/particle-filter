import numpy as np
from particle_filter import ParticleFilter
from prob_utils import sample
from measurement_model import measurement_likelihood
from motion_model import motion_model
from data_utils import DataLoader, GroundTruth, Control
from plotting import Plotter
from state import State

# USER PARAMS
N_PARTICLES = 2
SIGMA_INITIAL_STATE = 1


def initialize_random_state(x0=0, y0=0, h0=0):
    # generate a state with a Gaussian distibution around the initial point
    state = [[x0+sample(SIGMA_INITIAL_STATE), y0+sample(SIGMA_INITIAL_STATE),
              h0+sample(SIGMA_INITIAL_STATE)] for _ in range(N_PARTICLES)]
    return state


if __name__ == "__main__":

    dl = DataLoader()

    # get the robot's starting position in world coordinates
    x0, y0, h0 = dl.ground_truth[0, GroundTruth.X], dl.ground_truth[0,
                                                                    GroundTruth.Y], dl.ground_truth[0, GroundTruth.H]

    state = initialize_random_state(x0, y0, h0)

    # instance the particle filter
    filter = ParticleFilter(motion_model=motion_model, measurement_likelihood=measurement_likelihood,
                            n_particles=N_PARTICLES, initial_state=state)

    # instance plotter
    plot = Plotter(N_PARTICLES, dl.landmarks,
                   Plotter.PLOT_FINAL_PATH)

    control = dl.control
    positions = np.zeros((len(control), N_PARTICLES, State.LENGTH))
    means = np.zeros((len(control), 3))
    previous_timestamp = None

    # run over the odometry data
    for i, row in enumerate(control):
        # first time check
        if previous_timestamp is None:
            previous_timestamp = row[Control.T]
        else:
            # get time in between timesteps
            dt = row[Control.T] - previous_timestamp
            # get measurements that happened between this and the previous timestep
            z = dl.get_measurements(previous_timestamp, row[Control.T])
            # get the ground truth measurements between our previous loop and this one
            tru = dl.get_groundtruth(previous_timestamp, row[Control.T])

            # call our particle filter to get new state
            state = filter.update(row, z, dt, dl.landmarks)

            # add the means to our mean-tracking matrix
            means[i, :] = np.array(
                [np.mean(state[:, State.X]), np.mean(state[:, State.Y]), np.mean(state[:, State.HEADING])])
            # update the plot (if configured for real time plotting)
            plot.update(means, state, z, tru, dl.landmarks, i)
            # update timestamp
            previous_timestamp = row[Control.T]

    plot.plot(means, dl.ground_truth)
