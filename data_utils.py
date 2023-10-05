import numpy as np

PATH_ODOM = 'ds1/ds1_Odometry.dat'
PATH_GROUNDTRUTH = 'ds1/ds1_Groundtruth.dat'
PATH_LANDMARKS = 'ds1/ds1_Landmark_Groundtruth.dat'
PATH_MEASUREMENTS = 'ds1/ds1_Measurement.dat'
PATH_BARCODES = 'ds1/ds1_Barcodes.dat'

# subject id, barcode id
BARC_S, BARC_B = 0, 1
# landmarks subject id, x coordinate, y coordinate
LMT_S, LMT_X, LMT_Y = 0, 1, 2
# ground truth timestamp, x coordinate, y coordinate, heading
GT_T, GT_X, GT_Y, GT_H = 0, 1, 2, 3
# odometry timestamp, velocity, angular velocity
ODOM_T, ODOM_V, ODOM_W = 0, 1, 2
# measurements timestamp, subject (barcode) id, range, bearing
MEAS_T, MEAS_S, MEAS_R, MEAS_B = 0, 1, 2, 3


class DataLoader:
    def __init__(self):
        # load data
        self.odometry = np.loadtxt(PATH_ODOM)
        self.ground_truth = np.loadtxt(PATH_GROUNDTRUTH)
        self.landmarks = np.loadtxt(PATH_LANDMARKS)
        self.measurements = np.loadtxt(PATH_MEASUREMENTS)
        self.barcodes = np.loadtxt(PATH_BARCODES)
        # this generates a list of the barcode IDs that are actually landmarks
        # needed because some barcodes are not landmarks, and are detected in our measurements
        self.barcodes_landmarks = self.barcodes[np.isin(
            self.barcodes[:, BARC_S], self.landmarks[:, LMT_S])][:, BARC_B]

    def get_measurements(self, timestamp1, timestamp2):
        """Searches for the measurements between the given timestamps,
            and filters out the ones that are not landmarks

        """
        indexes = np.where((timestamp1 < self.measurements[:, MEAS_T]) &
                           (self.measurements[:, MEAS_T] <= timestamp2) &
                           (np.isin(self.measurements[:, BARC_B], self.barcodes_landmarks)))  # indexes of measurements between the given timesteps and that are landmarks
        x = self.measurements[indexes, :][0].copy()
        # replace the barcode ID with the landmark ID
        for i in range(x.shape[0]):
            x[i, MEAS_S] = self.barcodes[np.where(
                self.barcodes[:, BARC_B] == x[i, MEAS_S])[0][0]][BARC_S]
        return x

    def get_groundtruth(self, prev_timestamp, this_timestamp):
        # return the ground truth measurements between the previous timestep and this one
        return self.ground_truth[np.where((prev_timestamp < self.ground_truth[:, GT_T]) & (self.ground_truth[:, GT_T] <= this_timestamp)), :][0]
