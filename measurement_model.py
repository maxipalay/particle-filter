import numpy as np
from state import State
from data_utils import Landmark, Measurement

SCALING_FACTOR = 10


def measurement_model(state, landmark):
    """Measurement model.
        Given a current state and a known landmark, 
        this function calculates the predicted measurement.
        Inputs:
            :state: current state
            :landmark: any known landmark
        Output:
            predicted measurement in the format [radius, theta]
    """
    # just for ease of understanding explicitly declare the variables
    xr = state[State.X]
    yr = state[State.Y]
    thetar = state[State.HEADING]
    xl = landmark[Landmark.X]
    yl = landmark[Landmark.Y]

    r = np.sqrt((xl-xr)**2+(yl-yr)**2)
    alpha = np.arctan2(yl-yr, xl-xr)    # alpha is global
    theta = alpha - thetar   # theta is relative wrt the robot
    return r, theta


def measurement_likelihood(z, state, map):
    """Measurement likelihood.
        Given a measurement, the current state and the map,
        this function returns a score for the likelihood
        that the landmark was actually observed. It is the "weight"
        function in the particle filter.
        Inputs:
            :z: list of landmarks measured
            :state: current state
            :map: list of known landmarks/map
        Returns:
            Score based on the likelihood that the measurement is correct.

        Note:
        This returns a score, which is not bounded and is not a probability.
    """
    probs = np.array([])
    for m in z:  # for each measurement
        # get the landmark truth
        landmark_truth = map[np.where(
            m[Measurement.SUBJECT] == map[:, Landmark.SUBJECT])][0]

        # get the measurement's range and bearing
        meas_r, meas_theta = m[Measurement.R], m[Measurement.B]

        # get a prediction from out measurement model
        pred_r, pred_theta = measurement_model(state, landmark_truth)

        # decompose the measurement and prediction into the x,y coordinates
        pred_x = pred_r * np.cos(pred_theta)
        pred_y = pred_r * np.sin(pred_theta)

        meas_x = meas_r * np.cos(meas_theta)
        meas_y = meas_r * np.sin(meas_theta)

        # range distance between measurement and prediction
        distance_r = np.sqrt((pred_x-meas_x)**2+(pred_y-meas_y)**2)
        # difference between predicted and measured bearing
        diff = meas_theta - pred_theta
        #diff = diff-2*np.pi if diff >= np.pi else diff
        #diff = diff+2*np.pi if diff < -np.pi else diff
        distance_b = abs(diff) * \
            SCALING_FACTOR 
        distance = distance_r*distance_b
        # append difference to scores list
        probs = np.append(probs, distance)

    return 1.0/np.prod(probs)
