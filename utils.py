import numpy as np
ALPHA_V = 0.1
ALPHA_W = 2.0

CONTROL_V, CONTROL_W = 0, 1
STATE_X, STATE_Y, STATE_THETA = 0, 1, 2
LMT_S, LMT_X, LMT_Y = 0, 1, 2

def sample(b):
    """Returns a sample form a Gaussian distirbution as described in
    ......
    """
    return 0.5*np.sum([np.random.uniform(-np.sqrt(b), np.sqrt(b)) for _ in range(12)])

def low_variance_sampler(state, weights):
    """Low variance sampler implemented following the
        Provided scheme in ...
    
    """
    x_hat = np.zeros(state.shape)
    # state.shape[0] is the number of samples
    r = np.random.uniform(0.0, 1.0/state.shape[0])
    c = weights[0]
    i = 0
    for m in range(state.shape[0]):
        U = r + (m)/float(state.shape[0])
        while U > c:
            i += 1
            c = c + weights[i]
        x_hat[m,:] = state[i]
    return x_hat



