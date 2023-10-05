import numpy as np
from prob_utils import sample

CONTROL_V, CONTROL_W = 0, 1 # control is represented by an array of [velocity, angular velocity] commands in [m/s,rad/s]
STATE_X, STATE_Y, STATE_THETA = 0, 1, 2 # state is represented by an array if size [n particles, 3]
# noise constants
SIGMA_V = 0.1
SIGMA_W = 2.0

def motion_model(state, control, dt):
    v = control[CONTROL_V]
    w = control[CONTROL_W]
    x0 = state[STATE_X]
    y0 = state[STATE_Y]
    theta0 = state[STATE_THETA]

    # add gaussian noise to the commands
    v = v + sample(SIGMA_V)
    w = w + sample(SIGMA_W)

    
    x = x0 + v * np.cos(theta0) * dt
    y = y0 + v * np.sin(theta0) * dt
    theta = theta0 + w * dt

    return np.array([x,y,theta])