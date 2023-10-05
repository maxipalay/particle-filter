import numpy as np
from state import State
from data_utils import Landmark, Measurement
SCALING_FACTOR = 10

# :measurement_model: a function that given a current state and a landmark, returns the expected measurement


def measurement_model(state, landmark):
    # Measurement model
    # Function that returns a landmark's relative location to the robot
    # Inputs:
    #   - robot x position wrt world
    #   - robot y position wrt world
    #   - robot theta wrt world
    #   - landmark x wrt world
    #   - landmark y wrt world
    # Outputs:
    #   - radius, theta wrt robot frame
    xr = state[State.X]
    yr = state[State.Y]
    thetar = state[State.HEADING]
    xl = landmark[Landmark.X]
    yl = landmark[Landmark.Y]

    r = np.sqrt((xl-xr)**2+(yl-yr)**2)
    # this is absolute orientation (wrt world frame)
    alpha = np.arctan2(yl-yr, xl-xr)
    theta = alpha - thetar   # this is relative orientation (wrt robot frame)
    return r, theta


def measurement_likelihood(z, state, map):
    """
        inputs:
        z - list of landmarks measured
        x - state
    """
    probs = np.array([])
    for m in z:  # for each measurement
        # get the landmark truth
        landmark_truth = map[np.where(m[Measurement.SUBJECT] == map[:, Landmark.SUBJECT])][0]
        # print(landmark_truth)
        # print(m)
        # print(landmark_truth)
        # get the landmark's x and y position in the map

        # get the measurement's range and bearing
        meas_r, meas_theta = m[Measurement.R], m[Measurement.B]
        # calculate r hat
        pred_r, pred_theta = measurement_model(state, landmark_truth)

        pred_x = pred_r * np.cos(pred_theta)
        pred_y = pred_r * np.sin(pred_theta)

        meas_x = meas_r * np.cos(meas_theta)
        meas_y = meas_r * np.sin(meas_theta)

        # distance goes from zero to ~max distance of sensor, di
        # meters, max is max range of sensor ~say 10
        distance_r = np.sqrt((pred_x-meas_x)**2+(pred_y-meas_y)**2)
        distance_b = np.abs(meas_theta - pred_theta) * \
            SCALING_FACTOR  # radians max is 2*pi
        distance = distance_r*distance_b
        probs = np.append(probs, distance)
    # print(np.prod(probs))
    return 1.0/np.prod(probs)
