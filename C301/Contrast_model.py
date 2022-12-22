from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import umap
import scipy.io
import os

def main():
    C301_feature = scipy.io.loadmat("c301_x.mat")
    c301_x = C301_feature['c301_x']
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    for n_components in range(2, 11):                                    # Adjust dimensions of output
        print(n_components)

        # The method of KPCA for dimension reduction
        Kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.5)
        KPCA_result = Kpca.fit_transform(c301_x)
        KPCA_result = scaler.fit_transform(KPCA_result)
        dataNew = 'Different_dimension\\KPCA_{}d.mat'
        scipy.io.savemat(dataNew.format(n_components), {'new_data2': KPCA_result})


        # The method of PCA for dimension reduction
        pca = PCA(n_components=n_components)
        PCA_result = pca.fit_transform(c301_x)
        PCA_result = scaler.fit_transform(PCA_result)
        dataNew = 'Different_dimension\\PCA_{}d.mat'
        scipy.io.savemat(dataNew.format(n_components), {'new_data2': PCA_result})


        # The method of UMAP for dimension reduction
        Umap = umap.UMAP(n_neighbors=15, n_components=n_components)
        Umap_result = Umap.fit_transform(c301_x)
        Umap_result = scaler.fit_transform(Umap_result)
        dataNew = 'Different_dimension\\Umap_{}d.mat'
        scipy.io.savemat(dataNew.format(n_components), {'new_data2': Umap_result})


        # The method of Isomap for dimension reduction
        embedding = Isomap(n_components = n_components)
        Isomap_result = embedding.fit_transform(c301_x)
        Isomap_result = scaler.fit_transform(Isomap_result)
        dataNew = 'Different_dimension\\Isomap_{}d.mat'
        scipy.io.savemat(dataNew.format(n_components), {'new_data2': Isomap_result})


    for n_components in range(2, 4):
        print(n_components)

        # The method of TSNE for dimension reduction
        tsne = TSNE(n_components = n_components)
        TSNE_result = tsne.fit_transform(c301_x)
        TSNE_result = scaler.fit_transform(TSNE_result)
        dataNew = 'Different_dimension\\TSNE_{}d.mat'
        scipy.io.savemat(dataNew.format(n_components), {'new_data2': TSNE_result})

    return 0

if __name__=='__main__':
    main()