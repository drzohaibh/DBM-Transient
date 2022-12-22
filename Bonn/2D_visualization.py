import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import textwrap
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def plot_2D_5_case(c, Method, non_sei, sei, df_case):
    plt.figure(figsize=(10, 6), dpi = 500)
    plt.suptitle(c,fontsize=22)
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=0.1)
    ax1 = plt.subplot(gs[0, :2])

    X = df_case[[c + ' ' + Method[0] + 'F1', c + ' ' + Method[0] + 'F2']]
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])

    plt.title('DBM_' + Method[0], fontsize=18)
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=20)
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=20)
    plt.xticks([])
    plt.yticks([])
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)


    ax2 = plt.subplot(gs[0, 2:4])

    X = df_case[[c + ' ' + Method[1] + 'F1', c + ' ' + Method[1] + 'F2']]
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])

    plt.title('DBM_' + Method[1], fontsize=18)
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=20)
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=20)
    plt.xticks([])
    plt.yticks([])
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)


    ax3 = plt.subplot(gs[0, 4:6])
    X = df_case[[c + ' ' + Method[2] + 'F1', c + ' ' + Method[2] + 'F2']]
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])

    plt.title(Method[2], fontsize=18)
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=20)
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=20)
    plt.xticks([])
    plt.yticks([])
    ax3.spines['right'].set_linewidth(2)
    ax3.spines['top'].set_linewidth(2)
    ax3.spines['bottom'].set_linewidth(2)
    ax3.spines['left'].set_linewidth(2)

    ax4 = plt.subplot(gs[1, :2])
    X = df_case[[c + ' ' + Method[3] + 'F1', c + ' ' + Method[3] + 'F2']]
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])

    plt.title(Method[3], fontsize=18)
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=20)
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=20)
    plt.xticks([])
    plt.yticks([])
    ax4.spines['right'].set_linewidth(2)
    ax4.spines['top'].set_linewidth(2)
    ax4.spines['bottom'].set_linewidth(2)
    ax4.spines['left'].set_linewidth(2)


    ax5 = plt.subplot(gs[1, 2:4])
    X = df_case[[c + ' ' + Method[4] + 'F1', c + ' ' + Method[4] + 'F2']]
    y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
    standardScaler = StandardScaler().fit(X)
    X_standard = standardScaler.transform(X)
    svc = LinearSVC(C=1)
    svc.fit(X_standard, y)
    x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
    y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
    plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
    plt.title(Method[4], fontsize=18)
    plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=20 , label = 'Non-seizure')
    plt.scatter(X_standard[non_sei:non_sei + sei, 0], X_standard[non_sei:non_sei + sei, 1], c='r', s=20, label = 'Seizure')
    plt.xticks([])
    plt.yticks([])
    ax5.spines['right'].set_linewidth(2)
    ax5.spines['top'].set_linewidth(2)
    ax5.spines['bottom'].set_linewidth(2)
    ax5.spines['left'].set_linewidth(2)
    legend = plt.legend(bbox_to_anchor=(1.10,0.5), loc="center left", borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.show()

def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid( np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1,1), np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1,1) )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['tab:green', 'tab:red'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap, alpha=0.25)

def plot_2D_6(case):
    plt.figure(figsize=(15, 12), dpi=500)
    plt.suptitle('Different methods on Bonn Dataset', fontsize=22)
    gs = gridspec.GridSpec(7, 7)
    gs.update(wspace=0.1)
    sub_title = ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5', 'Case 6', 'Case 7']
    for i in range(len(case)):
        c = case[i]
        if c == 'case7': non_sei, sei = 400, 100          # number of non_seizure and seizure
        elif c == 'case5' or c == 'case6': non_sei, sei = 300, 100
        else: non_sei, sei = 100, 100
        Method=['transient', 'converged', 'KPCA', 'Isomap', 'TSNE', 'Umap'] # Method
        Method_title = ['transient', 'converged', 'KPCA', 'Isomap', 't-SNE', 'UMAP']
        df_case = pd.DataFrame()
        for M in Method:
            dir = 'Different_dimension\{}_{}_2d.mat'
            matrix = scipy.io.loadmat(dir.format(c, M))
            main_data = np.array(matrix['new_data2'])
            df_case[c + ' ' + M + '-F1'] = main_data[:, 0]
            df_case[c + ' ' + M + '-F2'] = main_data[:, 1]

        ax1 = plt.subplot(gs[1, i])
        ax1.set_title(sub_title[i], fontsize=18)
        X = df_case[[c + ' ' + Method[0] + '-F1', c + ' ' + Method[0] + '-F2']]
        y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
        standardScaler = StandardScaler().fit(X)
        X_standard = standardScaler.transform(X)
        svc = LinearSVC(C=1)
        svc.fit(X_standard, y)
        x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
        y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
        plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])

        if i == 0:
            ax1.set_ylabel('DBM_' + Method_title[0],fontsize=10, fontweight='bold')

        plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=7)
        plt.scatter(X_standard[non_sei:non_sei+sei,0],X_standard[non_sei:non_sei+sei,1], c='r', s=8)
        plt.xticks([])
        plt.yticks([])
        ax1.spines['right'].set_linewidth(2)
        ax1.spines['top'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.spines['left'].set_linewidth(2)


        ax2 = plt.subplot(gs[2, i])
        X = df_case[[c + ' ' + Method[1] + '-F1', c + ' ' + Method[1] + '-F2']]
        y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
        standardScaler = StandardScaler().fit(X)
        X_standard = standardScaler.transform(X)
        svc = LinearSVC(C=1)
        svc.fit(X_standard, y)
        x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
        y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
        plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])

        if i == 0:
            ax2.set_ylabel('DBM_' + Method_title[1], fontsize=10, fontweight='bold')
        plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=7)
        plt.scatter(X_standard[non_sei:non_sei+sei,0],X_standard[non_sei:non_sei+sei,1], c='r', s=8)
        plt.xticks([])
        plt.yticks([])
        ax2.spines['right'].set_linewidth(2)
        ax2.spines['top'].set_linewidth(2)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.spines['left'].set_linewidth(2)


        ax3 = plt.subplot(gs[3, i])
        X = df_case[[c + ' ' + Method[2] + '-F1', c + ' ' + Method[2] + '-F2']]
        y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
        standardScaler = StandardScaler().fit(X)
        X_standard = standardScaler.transform(X)
        svc = LinearSVC(C=1)
        svc.fit(X_standard, y)
        x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
        y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
        plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
        if i == 0:
            ax3.set_ylabel(Method_title[2], fontsize=10, fontweight='bold')
        plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=7)
        plt.scatter(X_standard[non_sei:non_sei+sei,0],X_standard[non_sei:non_sei+sei,1], c='r', s=8)
        plt.xticks([])
        plt.yticks([])
        ax3.spines['right'].set_linewidth(2)
        ax3.spines['top'].set_linewidth(2)
        ax3.spines['bottom'].set_linewidth(2)
        ax3.spines['left'].set_linewidth(2)


        ax4 = plt.subplot(gs[4, i])
        X = df_case[[c + ' ' + Method[3] + '-F1', c + ' ' + Method[3] + '-F2']]
        y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
        standardScaler = StandardScaler().fit(X)
        X_standard = standardScaler.transform(X)
        svc = LinearSVC(C=1)
        svc.fit(X_standard, y)
        x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
        y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
        plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
        if i == 0:
            ax4.set_ylabel(Method_title[3], fontsize=10, fontweight='bold')
        plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=7)
        plt.scatter(X_standard[non_sei:non_sei+sei,0],X_standard[non_sei:non_sei+sei,1], c='r', s=8)
        plt.xticks([])
        plt.yticks([])
        ax4.spines['right'].set_linewidth(2)
        ax4.spines['top'].set_linewidth(2)
        ax4.spines['bottom'].set_linewidth(2)
        ax4.spines['left'].set_linewidth(2)


        ax5 = plt.subplot(gs[5, i])
        X = df_case[[c + ' ' + Method[4] + '-F1', c + ' ' + Method[4] + '-F2']]
        y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
        standardScaler = StandardScaler().fit(X)
        X_standard = standardScaler.transform(X)
        svc = LinearSVC(C=1)
        svc.fit(X_standard, y)
        x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
        y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
        plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
        if i == 0:
            ax5.set_ylabel(Method_title[4], fontsize=10, fontweight='bold')
        plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=7, label='Non-Seizure')
        plt.scatter(X_standard[non_sei:non_sei+sei,0],X_standard[non_sei:non_sei+sei,1], c='r', s=8, label='Seizure')
        plt.xticks([])
        plt.yticks([])
        ax5.spines['right'].set_linewidth(2)
        ax5.spines['top'].set_linewidth(2)
        ax5.spines['bottom'].set_linewidth(2)
        ax5.spines['left'].set_linewidth(2)


        ax6 = plt.subplot(gs[6, i])
        X = df_case[[c + ' ' + Method[5] + '-F1', c + ' ' + Method[5] + '-F2']]
        y = np.row_stack((np.zeros((non_sei, 1)), np.ones((sei, 1))))
        standardScaler = StandardScaler().fit(X)
        X_standard = standardScaler.transform(X)
        svc = LinearSVC(C=1)
        svc.fit(X_standard, y)
        x_min, x_max = X_standard[:, 0].min() - 0.1, X_standard[:, 0].max() + 0.1
        y_min, y_max = X_standard[:, 1].min() - 0.1, X_standard[:, 1].max() + 0.1
        plot_decision_boundary(svc, axis=[x_min, x_max, y_min, y_max])
        if i == 0:
            ax6.set_ylabel(Method_title[5], fontsize=10, fontweight='bold')
        plt.scatter(X_standard[0:non_sei, 0], X_standard[0:non_sei, 1], c='g', s=7, label='Non-Seizure')
        plt.scatter(X_standard[non_sei:non_sei+sei,0],X_standard[non_sei:non_sei+sei,1], c='r', s=8, label='Seizure')
        plt.xticks([])
        plt.yticks([])
        ax6.spines['right'].set_linewidth(2)
        ax6.spines['top'].set_linewidth(2)
        ax6.spines['bottom'].set_linewidth(2)
        ax6.spines['left'].set_linewidth(2)
    plt.legend(loc='upper right', bbox_to_anchor=(-1.5, -0.05),fancybox = True, shadow = True, ncol = 2, fontsize=12)
    plt.show(bbox_inches='tight')


def main(c):
    case = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7']
    if c == 'case7': non_sei, sei = 400, 100                    # number of non_seizure and seizure
    elif c == 'case5' or c == 'case6': non_sei, sei = 300, 100
    else: non_sei, sei = 100, 100
    Method=['transient', 'converged', 'KPCA', 'Isomap', 'TSNE', 'UMAP'] # Methods
    df_case = pd.DataFrame()
    for M in Method:
        dir = 'Different_dimension\{}_{}_2d.mat'
        matrix = scipy.io.loadmat(dir.format(c, M))
        main_data = np.array(matrix['new_data2'])
        df_case[c + ' ' + M + 'F1'] = main_data[:, 0]
        df_case[c + ' ' + M + 'F2'] = main_data[:, 1]
    plot_2D_5_case(c, Method, non_sei, sei, df_case)
    plot_2D_6(case)


if __name__ == '__main__':
    case = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7']
    c = case[0]                                     # change the case
    main(c)