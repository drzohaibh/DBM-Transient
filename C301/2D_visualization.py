import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid( np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1,1), np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1,1) )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['tab:green', 'tab:red'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap, alpha=0.15)

def plot_2D_6(M, non_sei, sei):
    dir = "Different_dimension\{}_2d.mat"
    Method_title = ['DBM_transient', 'DBM_converged', 'KPCA', 'Isomap', 't-SNE', 'UMAP']
    plt.figure(figsize=(10, 6), dpi=500)
    plt.suptitle('Different methods on C301 Dataset', fontsize=22)
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=0.1)
    ax1 = plt.subplot(gs[0, 0:2])
    plt.title(Method_title[0],fontsize=12, fontweight='bold')
    matrix = scipy.io.loadmat(dir.format(M[0]))
    X = np.array(matrix['new_data2'])
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=5)
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=5)
    plt.xticks([])
    plt.yticks([])
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)


    ax2 = plt.subplot(gs[0, 2:4])
    plt.title(Method_title[1],fontsize=12, fontweight='bold')
    matrix = scipy.io.loadmat(dir.format(M[1]))
    X = np.array(matrix['new_data2'])
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=5)
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=5)
    plt.xticks([])
    plt.yticks([])
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)

    ax3 = plt.subplot(gs[0, 4:6])
    plt.title(Method_title[2],fontsize=12, fontweight='bold')
    matrix = scipy.io.loadmat(dir.format(M[2]))
    X = np.array(matrix['new_data2'])
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=5)
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=5)
    plt.xticks([])
    plt.yticks([])
    ax3.spines['right'].set_linewidth(2)
    ax3.spines['top'].set_linewidth(2)
    ax3.spines['bottom'].set_linewidth(2)
    ax3.spines['left'].set_linewidth(2)

    ax4 = plt.subplot(gs[1, 0:2])
    plt.title(Method_title[3],fontsize=12, fontweight='bold')
    matrix = scipy.io.loadmat(dir.format(M[3]))
    X = np.array(matrix['new_data2'])
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=5)
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=5)
    plt.xticks([])
    plt.yticks([])
    ax4.spines['right'].set_linewidth(2)
    ax4.spines['top'].set_linewidth(2)
    ax4.spines['bottom'].set_linewidth(2)
    ax4.spines['left'].set_linewidth(2)

    ax5 = plt.subplot(gs[1, 2:4])
    plt.title(Method_title[4],fontsize=12, fontweight='bold')
    matrix = scipy.io.loadmat(dir.format(M[4]))
    X = np.array(matrix['new_data2'])
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=5, label='Non-Seizure')
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=5, label='Seizure')
    plt.xticks([])
    plt.yticks([])
    ax5.spines['right'].set_linewidth(2)
    ax5.spines['top'].set_linewidth(2)
    ax5.spines['bottom'].set_linewidth(2)
    ax5.spines['left'].set_linewidth(2)

    ax6 = plt.subplot(gs[1, 4:6])
    plt.title(Method_title[5],fontsize=12, fontweight='bold')
    matrix = scipy.io.loadmat(dir.format(M[5]))
    X = np.array(matrix['new_data2'])
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=5, label='Non-Seizure')
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=5, label='Seizure')
    plt.xticks([])
    plt.yticks([])
    ax6.spines['right'].set_linewidth(2)
    ax6.spines['top'].set_linewidth(2)
    ax6.spines['bottom'].set_linewidth(2)
    ax6.spines['left'].set_linewidth(2)
    plt.legend(loc='upper left', bbox_to_anchor=(-1.17, -0.05),fancybox = True, shadow = True, ncol = 2, fontsize=12)
    plt.show(bbox_inches='tight')


def main():

    non_sei, sei = 9992, 757
    Method=['DBM_transient', 'DBM_converged', 'KPCA', 'Isomap', 'TSNE', 'Umap'] # Methods
    df_301 = pd.DataFrame()
    for M in Method:
        dir = "Different_dimension\{}_2d.mat"
        matrix = scipy.io.loadmat(dir.format(M))
        main_data = np.array(matrix['new_data2'])
        df_301[M + 'F1'] = main_data[:, 0]
        df_301[M + 'F2'] = main_data[:, 1]
    plot_2D_6(Method, non_sei, sei)

if __name__ == '__main__':
    main()