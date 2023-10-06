import numpy as np
from particle_filter import ParticleFilter
from prob_utils import sample
from measurement_model import measurement_likelihood
from motion_model import motion_model
from data_utils import DataLoader, GroundTruth, Control
from plotting import Plotter
from state import State

# USER PARAMS
N_PARTICLES = 100
SIGMA_INITIAL_STATE = 1
SMOOTH_MEAN = False
SMOOTHING_ALPHA = 0.8

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
    means = np.zeros((len(control)-1, State.LENGTH))
    distance_position = []
    distance_heading = []
    previous_timestamp = None

    valid_means_counter = 0
    # run over the odometry data
    for row in control:
        # first time check
        if previous_timestamp is None:
            previous_timestamp = row[Control.T]
        else:
            # get time in between timesteps
            dt = row[Control.T] - previous_timestamp
            # get measurements that happened between this and the previous timestep
            z = dl.get_measurements(previous_timestamp, row[Control.T])
            # get the ground truth measurements between our previous loop and this one
            truth = dl.get_groundtruth(previous_timestamp, row[Control.T])

            # call our particle filter to get new state
            state = filter.update(row, z, dt, dl.landmarks)

            # add the means to our mean-tracking matrix
            if SMOOTH_MEAN and valid_means_counter > 1:
                means[valid_means_counter, :] = np.array(
                    [means[valid_means_counter-1,State.X]*SMOOTHING_ALPHA+np.mean(state[:, State.X])*(1-SMOOTHING_ALPHA), 
                    means[valid_means_counter-1,State.Y]*SMOOTHING_ALPHA+np.mean(state[:, State.Y])*(1-SMOOTHING_ALPHA),
                    means[valid_means_counter-1,State.HEADING]*SMOOTHING_ALPHA+np.mean(state[:, State.HEADING])*(1-SMOOTHING_ALPHA)])
            else:
                means[valid_means_counter, :] = np.array([np.mean(state[:, State.X]), 
                    np.mean(state[:, State.Y]),
                    np.mean(state[:, State.HEADING])])
            
            # add the difference between the predicted
            if truth.size > 0:
                distance_position.append(np.sqrt((means[valid_means_counter, State.X]-np.mean(truth[:, GroundTruth.X]))**2+(
                    means[valid_means_counter, State.Y]-np.mean(truth[:, GroundTruth.Y]))**2))
                
                mean_heading = means[valid_means_counter, State.HEADING]*180.0/np.pi%360
                # the heading isn't bounded
                mean_heading = mean_heading - 360.0 if mean_heading >= 180.0 else mean_heading
                truth_heading = np.mean(truth[:, GroundTruth.H])*180.0/np.pi
                # difference
                diff = mean_heading - truth_heading
                diff = diff-360.0 if diff >= 180.0 else diff
                diff = diff+360.0 if diff < -180.0 else diff
                diff = abs(diff)
                distance_heading.append(diff)

            plot.update(means, state, z, truth, dl.landmarks, valid_means_counter)
            valid_means_counter += 1
            previous_timestamp = row[Control.T]

    print(f"n_particles: {N_PARTICLES}")
    print(f"mean distance [meters]: {np.mean(distance_position)}")
    print(f"mean difference heading [degrees]: {np.mean(distance_heading)}")
    plot.plot(means[0:valid_means_counter-1], dl.ground_truth)
