import numpy as np
PATH_ODOM = 'ds1/ds1_Odometry.dat'
PATH_GROUNDTRUTH = 'ds1/ds1_Groundtruth.dat'
PATH_LANDMARKS = 'ds1/ds1_Landmark_Groundtruth.dat'
PATH_MEASUREMENTS = 'ds1/ds1_Measurement.dat'
PATH_BARCODES = 'ds1/ds1_Barcodes.dat'


class Landmark:
    """Pseudo-enum for indexing the Landmark vectors."""
    SUBJECT, X, Y = 0, 1, 2


class Measurement:
    """Pseudo-enum for indexing the Measurement vectors."""
    T, SUBJECT, R, B = 0, 1, 2, 3  # timestamp, subject id,


class Control:
    """Pseudo-enum for indexing the Control/Odometry vectors."""
    T, V, W = 0, 1, 2  # timestamp, vel, ang. vel.


class GroundTruth:
    """Pseudo-enum for indexing the GroundTruth vectors."""
    T, X, Y, H = 0, 1, 2, 3  # timestamp, x, y, heading


class Barcode:
    """Pseudo-enum for indexing the Barcode vectors."""
    S, B = 0, 1  # subject ID, barcode ID


class DataLoader:
    """
    Helper class to load the datasets into numpy arrays.
    """

    def __init__(self):
        # load data
        self.control = np.loadtxt(PATH_ODOM)
        self.ground_truth = np.loadtxt(PATH_GROUNDTRUTH)
        self.landmarks = np.loadtxt(PATH_LANDMARKS)
        self.measurements = np.loadtxt(PATH_MEASUREMENTS)
        self.barcodes = np.loadtxt(PATH_BARCODES)
        # this generates a list of the barcode IDs that are actually landmarks
        # needed because some barcodes are not landmarks, and are detected in our measurements
        self.barcodes_landmarks = self.barcodes[np.isin(
            self.barcodes[:, Barcode.S], self.landmarks[:, Landmark.SUBJECT])][:, Barcode.B]

    def get_measurements(self, timestamp1, timestamp2):
        """Searches for the measurements between the given timestamps, filters out the ones that are not landmarks, and replaces the measurement ID with a landmark ID"""
        # indexes of measurements between the given timesteps and that are landmarks (we're filtering out measurements whose subject id is not a landmark id)
        indexes = np.where((timestamp1 < self.measurements[:, Measurement.T]) &
                           (self.measurements[:, Measurement.T] <= timestamp2) &
                           (np.isin(self.measurements[:, Barcode.B], self.barcodes_landmarks)))
        x = self.measurements[indexes, :][0].copy()
        # replace the barcode ID (=measurement ID) with the landmark ID
        for i in range(x.shape[0]):
            x[i, Measurement.SUBJECT] = self.barcodes[np.where(
                self.barcodes[:, Barcode.B] == x[i, Measurement.SUBJECT])[0][0]][Barcode.S]
        return x

    def get_groundtruth(self, prev_timestamp, this_timestamp):
        """Returns the ground truth measurements between the given timesteps."""
        return self.ground_truth[np.where((prev_timestamp < self.ground_truth[:, GroundTruth.T]) & (self.ground_truth[:, GroundTruth.T] <= this_timestamp)), :][0]
