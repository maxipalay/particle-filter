import numpy as np
ALPHA_V = 0.1
ALPHA_W = 2.0

CONTROL_V, CONTROL_W = 0, 1
STATE_X, STATE_Y, STATE_THETA = 0, 1, 2
LMT_S, LMT_X, LMT_Y = 0, 1, 2

from utils import sample

def motion_model(state, control, dt):
    v = control[CONTROL_V]
    w = control[CONTROL_W]
    x0 = state[STATE_X]
    y0 = state[STATE_Y]
    theta0 = state[STATE_THETA]

    # add gaussian noise to the commands
    v = v + sample(ALPHA_V)
    w = w + sample(ALPHA_W)

    
    x = x0 + v * np.cos(theta0) * dt
    y = y0 + v * np.sin(theta0) * dt
    theta = theta0 + w * dt

    return np.array([x,y,theta])