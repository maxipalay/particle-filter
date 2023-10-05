import numpy as np
from prob_utils import low_variance_sampler
from data_utils import Control
from state import State


class ParticleFilter:
    """Class representing a Particle Filter.
        Args:
            :motion_model: a function that given a state sample, a control and a dt, returns a new state sample
            :measurement_likelihood: a function that given a state sample and a list of landmarks, calculates a score for the likelihood of landmarks observation
            :initial_state: the state for the first run of the particle filter
            :n_particles: the number of particles to use
    """

    def __init__(self, motion_model, measurement_likelihood, initial_state, n_particles):
        self._n_particles = n_particles
        self._state = initial_state
        self._motion_model = motion_model
        self._measurement_likelihood = measurement_likelihood

    def update(self, control, measurements, dt, map):
        """Update the state.
            Input args:
                :control: current control inputs
                :measurements: list of current measurements
                :dt: time for which the controls are applied
            Output:
                new state
        """
        # if there is a control command or measurement, run the filter
        if control[Control.V] != 0.0 or control[Control.W] != 0.0 or measurements.size > 0:
            # initialize variables
            x, x_hat, weights = np.zeros((self._n_particles, State.LENGTH)), np.zeros(
                (self._n_particles, State.LENGTH)), np.zeros(self._n_particles)
            # for each particle
            for m in range(self._n_particles):
                # get a sample from the motion model
                temp_state = self._motion_model(self._state[m], control, dt)
                # calculate the weight for the sample
                weights[m] = self._measurement_likelihood(
                    measurements, temp_state, map)
                # save the sample in x_hat
                x_hat[m, :] = temp_state

            # normalize the weights
            weights /= np.sum(weights)

            # resample only if there are measurements
            if measurements.size > 0:
                self._state = low_variance_sampler(x_hat, weights).copy()
            else:
                self._state = x_hat.copy()

        return self._state
