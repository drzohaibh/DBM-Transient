from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import scipy.io
import numpy as np
import random
import umap
import os
import pandas as pd


def main(c):
    bonn_feature = scipy.io.loadmat("Bonn.mat")
    bonn_x = bonn_feature['trainx']                                     # Download the bonn feature
    case1 = np.row_stack((bonn_x[0:100, :], bonn_x[400:500, :]))        # select cases for training.........
    case2 = np.row_stack((bonn_x[100:200, :], bonn_x[400:500, :]))
    case3 = np.row_stack((bonn_x[200:300, :], bonn_x[400:500, :]))
    case4 = np.row_stack((bonn_x[300:400, :], bonn_x[400:500, :]))
    case5 = np.row_stack((bonn_x[0:100, :], bonn_x[200:500, :]))
    case6 = np.row_stack((bonn_x[100:400, :], bonn_x[400:500, :]))
    case7 = bonn_x
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_x = eval(c)                                                    # Make the training data equal to the case

    for n_components in range(2, 11):                                    # Adjust dimensions of output
        print(n_components)

        # The method of KPCA for dimension reduction
        Kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.5)
        KPCA_result = Kpca.fit_transform(train_x)
        KPCA_result = scaler.fit_transform(KPCA_result)
        dataNew = 'Different_dimension\\{}_KPCA_{}d.mat'
        scipy.io.savemat(dataNew.format(c, n_components), {'new_data2': KPCA_result})


        # The method of PCA for dimension reduction
        pca = PCA(n_components=n_components)
        PCA_result = pca.fit_transform(train_x)
        PCA_result = scaler.fit_transform(PCA_result)
        dataNew = 'Different_dimension\\{}_PCA_{}d.mat'
        scipy.io.savemat(dataNew.format(c, n_components), {'new_data2': PCA_result})


        # The method of Isomap for dimension reduction
        embedding = Isomap(n_components = n_components)
        Isomap_result = embedding.fit_transform(train_x)
        Isomap_result = scaler.fit_transform(Isomap_result)
        dataNew = 'Different_dimension\\{}_Isomap_{}d.mat'
        scipy.io.savemat(dataNew.format(c, n_components), {'new_data2': Isomap_result})


        # The method of UMAP for dimension reduction
        Umap = umap.UMAP(n_neighbors=15, n_components=n_components)
        Umap_result = Umap.fit_transform(train_x)  # fit the model
        Umap_result = scaler.fit_transform(Umap_result)
        dataNew = 'Different_dimension\\{}_Umap_{}d.mat'
        scipy.io.savemat(dataNew.format(c, n_components), {'new_data2': Umap_result})

    for n_components in range(2, 4):
        print(n_components)

        # The method of TSNE for dimension reduction
        tsne = TSNE(n_components = n_components)
        TSNE_result = tsne.fit_transform(train_x)
        TSNE_result = scaler.fit_transform(TSNE_result)
        dataNew = 'Different_dimension\\{}_TSNE_{}d.mat'
        scipy.io.savemat(dataNew.format(c, n_components), {'new_data2': TSNE_result})

    return 0


if __name__ == '__main__':
    case = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7']
    main(case[0])                                       # Change case[n] and select different cases for calculation
                                                        # n∈[0, 1, 2, 3, 4, 5, 6]