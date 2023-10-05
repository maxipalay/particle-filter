import numpy as np
from utils import sample, low_variance_sampler

CONTROL_V, CONTROL_W = 0, 1
####### Magic numbers in self._state initialization
###### more magic numbers (configuration_length)

class ParticleFilter:
    """Class representing a Particle Filter.
        Args:
            :motion_model: a function that given a state sample, a control and a dt, returns a new state sample
            :landmark_likelihood: a function that given a state sample and a list of landmarks, calculates a score for the likelihood of landmarks observation
            :n_particles: the number of particles to use
            :config_length:
    """
    def __init__(self, motion_model, landmark_likelihood, n_particles, config_length, initial_state = None):
        """Args:
            :measurement_model:
            :motion_model:
            :initial_state:
        """

        # if no initial state
        if initial_state is None:
            # initialize random gaussian state
            self._state = [[sample(1) for _ in range(config_length)] for _ in range(n_particles)]
        else:
            self._state = initial_state
        self._motion_model = motion_model
        self._n_particles = n_particles
        self._config_length = config_length
        self._landmark_likelihood = landmark_likelihood

    def update(self, control, measurements, dt):
        """Update our state.
            Input args:
                :control:
                :measurements:
                :dt:
        """
        # if there is a control command or measurement, run the filter
        if control[CONTROL_V] != 0.0 or control[CONTROL_W] != 0.0 or measurements.size > 0:
            # initialize variables
            x, x_hat, weights = np.zeros((self._n_particles, self._config_length)), np.zeros((self._n_particles,self._config_length)), np.zeros(self._n_particles)
            # for each particle
            for m in range(self._n_particles):
                # get a sample from the motion model
                temp_state = self._motion_model(self._state[m], control, dt)
                # calculate the weight for the sample
                weights[m] = self._landmark_likelihood(measurements, temp_state)
                # save the sample in x_hat
                x_hat[m,:] = temp_state

            # normalize the weights
            weights /= np.sum(weights)

            # resample only if there are measurements
            if measurements.size > 0:
                self._state = low_variance_sampler(x_hat, weights).copy()
            else:
                self._state = x_hat.copy()
                
        return self._state

    