import numpy as np
import math
import scipy.io
import os
import pywt

def data_Preprocess(train_set):
    C = []
    for i in range(0, len(train_set[:, 0])):
        c = []
        # calculate wavelet subband
        cA1, cD1 = pywt.dwt(train_set[i, :], 'db4')
        cA2, cD2 = pywt.dwt(cA1, 'db4')
        cA3, cD3 = pywt.dwt(cA2, 'db4')
        cA4, cD4 = pywt.dwt(cA3, 'db4')
        cA5, cD5 = pywt.dwt(cA4, 'db4')

        # with tf.variable_scope('wavelet_coef_set'):
        # coef_A   Mean of the absolute values of the coefficients in each sub-band.
        coef_A1 = sum(abs(cD3)) / len(cD3)
        coef_A2 = sum(abs(cD4)) / len(cD4)
        coef_A3 = sum(abs(cD5)) / len(cD5)
        coef_A4 = sum(abs(cA5)) / len(cA5)

        # coef_B   Average power of the wavelet coefficients in each sub-band.
        coef_B1 = sum(cD3 * cD3.T) / (len(cD3))
        coef_B2 = sum(cD4 * cD4.T) / (len(cD4))
        coef_B3 = sum(cD5 * cD5.T) / (len(cD5))
        coef_B4 = sum(cA5 * cA5.T) / (len(cA5))

        # coef_C  Standard deviation of the coefficients in each sub-band.
        coef_C1 = np.std(cD3, ddof=1)
        coef_C2 = np.std(cD4, ddof=1)
        coef_C3 = np.std(cD5, ddof=1)
        coef_C4 = np.std(cA5, ddof=1)

        # coef_D  Ratio of the absolute mean values of adjacent sub-bands.
        coef_D1 = (sum(abs(cD4)) / len(cD4)) / (sum(abs(cD3)) / len(cD3))
        coef_D2 = (sum(abs(cD5)) / len(cD5)) / (sum(abs(cD4)) / len(cD4))

        c = [coef_A1, coef_A2, coef_A3, coef_A4, coef_B1, coef_B2, coef_B3, coef_B4, coef_C1, coef_C2, coef_C3, coef_C4,
             coef_D1, coef_D2]
        c = np.array(c)
        C.append(c)
    C = np.array(C)
    X = np.zeros((len(C[:, 0]), len(C[0, :])))
    for i in range(len(C[0, :])):
        X_sub =  1 / (max(C[:, i]) - min(C[:, i])) * (C[:, i] - min(C[:, i]))                 #  Normalization between [0, 1]
        X[:, i] = X_sub
    return C, X

def main():

    dir = 'Different_dimension'
    if not os.path.exists(dir):
        os.mkdir(dir)

    data_path = "C301_data/patient_1-19.mat"
    patient_data = scipy.io.loadmat(data_path)
    sample_non, sample_sei = patient_data['sample_non'], patient_data['sample_sei']
    train_set = np.row_stack((sample_non, sample_sei))
    c301_feature, c301_x = data_Preprocess(train_set)
    dataNew = "c301_x.mat"
    scipy.io.savemat(dataNew, {'c301_x': c301_x})


if __name__ == '__main__':
    main()

