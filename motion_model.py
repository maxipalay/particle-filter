import numpy as np
from prob_utils import sample
from state import State
from data_utils import Control

# noise constants
SIGMA_V = 0.1
SIGMA_W = 2.0


def motion_model(state, control, dt):
    """Function that represents a motion model. 
        Input args:
            :state: current state
            :control: current control
            :dt: time for which the control is applied (dt)
        Output:
            Predicted state at time t+dt
    """
    # just for ease of understanding explicitly declare the variables
    v = control[Control.V]
    w = control[Control.W]
    x0 = state[State.X]
    y0 = state[State.Y]
    theta0 = state[State.HEADING]

    # add gaussian noise to the commands
    v = v + sample(SIGMA_V)
    w = w + sample(SIGMA_W)

    # the actual motion model
    x = x0 + v * np.cos(theta0) * dt
    y = y0 + v * np.sin(theta0) * dt

    # this can cause issues because theta is unbounded
    theta = theta0 + w * dt

    return np.array([x, y, theta])
