import numpy as np

BARC_S, BARC_B = 0, 1
LMT_S, LMT_X, LMT_Y = 0, 1, 2
GT_T, GT_X, GT_Y, GT_H = 0, 1, 2, 3
# indexes in files
ODOM_T, ODOM_V, ODOM_W = 0, 1, 2
MEAS_T, MEAS_S, MEAS_R, MEAS_B = 0, 1, 2, 3


# files to open
odometry = np.loadtxt('ds1/ds1_Odometry.dat')
ground_truth = np.loadtxt('ds1/ds1_Groundtruth.dat')
landmarks = np.loadtxt('ds1/ds1_Landmark_Groundtruth.dat')
measurements = np.loadtxt('ds1/ds1_Measurement.dat')
barcodes = np.loadtxt('ds1/ds1_Barcodes.dat')

# numbers of barcodes that are landmarks
barcodes_landmarks = barcodes[np.isin(barcodes[:,BARC_S],landmarks[:,LMT_S])][:,BARC_B]

def measure(prev_timestep, this_timestep):
    # return the measurements between the previous timestep and this one
    indexes = np.where((prev_timestep < measurements[:, MEAS_T]) & (measurements[:, MEAS_T] <= this_timestep) & (
        np.isin(measurements[:, BARC_B], barcodes_landmarks)))  # indexes of measurements between the given timesteps

    return measurements[indexes, :][0]


def groundtruth(prev_timestep, this_timestep):
        # return the measurements between the previous timestep and this one
        #print(tmeasurements[:,MEAS_T]))
        #print(type(prev_timestep))
        return ground_truth[np.where((prev_timestep < ground_truth[:,GT_T]) & (ground_truth[:,GT_T] <= this_timestep)),:][0]

def get_landmark(subject_number):
        x = landmarks[np.where(landmarks[:,0] == subject_number)[0],:][0]
        return x

def get_landmark_from_measurement(z):
    # The measurements capture the Barcode as the Subject #
    # this function translates the barcode number obtained from the measurements into 
    # the landmark groundtruth (x,y)
    #print(f"measurement: {z}")
    #
    #print(measurement)
    #  indexes of the elements in the barcodes matrix which barcode code equals to the measurement subject code
    idx = np.where(barcodes[:,BARC_B] == z[MEAS_S])
    # grab the number of the subject corresponding to that measurement code
    s = int(barcodes[idx, BARC_S][0][0])
    #print(f"landmark_subject {s}")
    # get the index of the row in landmarks truth of the corresponding landmark
    a = np.where(landmarks[:,LMT_S] == s)[0]
    if a.size>0:
        #print(output.shape)
        #print(landmarks[a,:].shape)
        output = np.array(landmarks[a,:][0])
        return output
    return np.array([])