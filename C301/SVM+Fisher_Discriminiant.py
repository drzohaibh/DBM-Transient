import numpy as np
import pandas as pd
import scipy.io
from sklearn import svm
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def Fisher(M, D, main_data):
    non_sei, sei = 9992, 757
    mean_A = main_data[0:non_sei, :].mean(axis=0)
    mean_E = main_data[non_sei:non_sei + sei, :].mean(axis=0)
    sw_1 = 0
    sw_2 = 0
    for i in range(non_sei):
        reducemean_A = main_data[i, :] - mean_A
        sw = np.matmul(reducemean_A[:, np.newaxis], reducemean_A[:, np.newaxis].T)
        sw_1 += sw
    sw_1 = np.divide(sw_1, non_sei)
    for i in range(sei):
        reducemean_E = main_data[i + non_sei, :] - mean_E
        sw = np.matmul(reducemean_E[:, np.newaxis], reducemean_E[:, np.newaxis].T)
        sw_2 += sw
    sw_2 = np.divide(sw_2, sei)
    sw = sw_1+sw_2
    w = np.matmul(np.linalg.inv(sw), (mean_E - mean_A)[:, np.newaxis])
    sb = np.matmul((mean_E - mean_A)[:, np.newaxis], (mean_E - mean_A)[:, np.newaxis].T)
    jw = np.matmul(np.matmul(w.T, sb), w) / np.matmul(np.matmul(w.T, sw), w)
    print('The J(w) of {} in {}D:'.format(M, D), jw[0][0])
    return jw[0][0]

def svm_train(M, D, main_data):
    non_sei, sei = 9992, 757
    n = main_data[0:non_sei, :]
    s = main_data[non_sei:non_sei + sei, :]
    dx_label = np.ones((len(s), 1))                      #Labels for seizure
    fdx_label = np.zeros((len(n), 1))                    #Labels for non-seizure
    train_rate = 0.7
    train_time = 100
    mitrix = np.zeros((train_time, 5))
    for i in range(train_time):
        shuffle_order1 = np.random.permutation(np.arange(len(s[:, 0])))
        X1 = s[shuffle_order1]
        Y1 = dx_label

        shuffle_order2 = np.random.permutation(np.arange(len(n[:, 0])))
        X2 = n[shuffle_order2]
        Y2 = fdx_label

        Train_X = np.row_stack((X1[0:int(len(X1[:, 0]) * train_rate), :], X2[0:int(len(X2[:, 0]) * train_rate), :]))
        Test_X = np.row_stack((X1[int(len(X1[:, 0]) * train_rate):len(X1[:, 0]), :],
                               X2[int(len(X2[:, 0]) * train_rate):len(X2[:, 0]), :]))

        Train_Y = np.row_stack((Y1[0:int(len(X1[:, 0]) * train_rate), :], Y2[0:int(len(X2[:, 0]) * train_rate), :]))
        Test_Y = np.row_stack((Y1[int(len(X1[:, 0]) * train_rate):len(X1[:, 0]), :],
                               Y2[int(len(X2[:, 0]) * train_rate):len(X2[:, 0]), :]))

        shuffle_order3 = np.random.permutation(np.arange(len(Train_X[:, 0])))
        Train_X = Train_X[shuffle_order3]
        Train_Y = Train_Y[shuffle_order3]

        train_data, test_data = Train_X, Test_X
        train_label, test_label = Train_Y[:, 0], Test_Y[:, 0]

        svm_classifier = svm.SVC(kernel='linear', C=1).fit(Train_X, train_label)

        predict_test = svm_classifier.predict(test_data)
        a = confusion_matrix(y_true=test_label, y_pred=predict_test)
        tn, fp, fn, tp = a.ravel()

        specificity = tn / (tn + fp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fn + fp)
        F1 = 2 * precision * recall / (recall + precision)

        mitrix[i, 0], mitrix[i, 1], mitrix[i, 2], mitrix[i, 3], mitrix[i, 4] = specificity, recall, precision, acc, F1
    mean_mitrix = np.mean(mitrix, axis=0)
    mean_mitrix = mean_mitrix[:, np.newaxis].T
    print('The Confusion matrix of {} in {}D:'.format(M, D), mean_mitrix)
    return mean_mitrix[0]

def main():
    Method=['DBM_transient', 'DBM_converged', 'KPCA', 'Isomap', 'TSNE', 'UMAP']
    df_JW = pd.DataFrame()
    df_matrix = pd.DataFrame()
    for M in Method:
        J = []
        if M != 'TSNE':
            print('******  The method of {} for calculate  ******'.format(M))
            for i in range(2, 11):
                matrix = scipy.io.loadmat("Different_dimension\{}_{}d.mat".format(M, str(i)))
                main_data = np.array(matrix['new_data2'])
                J.append(Fisher(M, i, main_data))
                df_matrix['{} {}D'.format(M, i)] = svm_train(M, i, main_data)
            df_JW[M] = J
            print('>>>>>>>>>>>>>>>>>>>>>> {} END <<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(M))
        else:
            print('******The method of {} for calculate******'.format(M))
            for i in range(2, 4):
                matrix = scipy.io.loadmat("Different_dimension\{}_{}d.mat".format(M, str(i)))
                main_data = np.array(matrix['new_data2'])
                J.append(Fisher(M, i, main_data))
                df_matrix['{} {}D'.format(M, i)] = svm_train(M, i, main_data)
            df2 = pd.DataFrame(J, index=['2D', '3D'], columns=[M])
            print('>>>>>>>>>>>>>>>>>>>>>> {} END <<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(M))
    df_JW.index = ['2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D']
    df_JW = pd.concat([df_JW, df2], axis=1)
    df_matrix.index = ['SPE', 'REC', 'PRE', 'ACC', 'F1']
    df_matrix.to_csv("Evaluation_results\C301_matrix.csv")
    df_JW.to_csv("Evaluation_results\C301_JW.csv")
    return df_JW, df_matrix

if __name__=='__main__':
    dir = 'Evaluation_results'
    if not os.path.exists(dir):
        os.mkdir(dir)

    df_JW, df_matrix = main()
