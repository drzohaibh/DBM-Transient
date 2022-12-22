import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def plot_bar(df_matrix):
    Method = ['DBM_transient', 'DBM_converged', 'KPCA', 'Isomap', 'TSNE', 'UMAP']
    D = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
    df_2d = pd.DataFrame()
    df_3d = pd.DataFrame()
    df_4d = pd.DataFrame()
    df_5d = pd.DataFrame()
    df_6d = pd.DataFrame()
    df_7d = pd.DataFrame()
    df_8d = pd.DataFrame()
    df_9d = pd.DataFrame()
    df_10d = pd.DataFrame()
    for M in Method:
        df_2d[M] = df_matrix[M + ' 2D']
        df_3d[M] = df_matrix[M + ' 3D']
        if M != 'TSNE':
            df_4d[M] = df_matrix[M + ' 4D']
            df_5d[M] = df_matrix[M + ' 5D']
            df_6d[M] = df_matrix[M + ' 6D']
            df_7d[M] = df_matrix[M + ' 7D']
            df_8d[M] = df_matrix[M + ' 8D']
            df_9d[M] = df_matrix[M + ' 9D']
            df_10d[M] = df_matrix[M + ' 10D']
    plt.figure()
    df_2d.plot.bar()
    plt.title('C301 {}D'.format(D[0]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(-0.09, -0.15), ncol = 5, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_2d.png")
    plt.show()

    plt.figure()
    df_3d.plot.bar()
    plt.title('C301 {}D'.format(D[1]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(-0.09, -0.15), ncol = 5, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_3d.png")
    plt.show()

    plt.figure()
    df_4d.plot.bar()
    plt.title('C301 {}D'.format(D[2]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.06, -0.15), ncol = 4, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_4d.png")
    plt.show()

    plt.figure()
    df_5d.plot.bar()
    plt.title('C301 {}D'.format(D[3]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.06, -0.15), ncol = 4, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_5d.png")
    plt.show()

    plt.figure()
    df_6d.plot.bar()
    plt.title('C301 {}D'.format(D[4]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.06, -0.15), ncol = 4, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_6d.png")
    plt.show()

    plt.figure()
    df_7d.plot.bar()
    plt.title('C301 {}D'.format(D[5]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.06, -0.15), ncol = 4, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_7d.png")
    plt.show()

    plt.figure()
    df_8d.plot.bar()
    plt.title('C301 {}D'.format(D[6]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.06, -0.15), ncol = 4, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_8d.png")
    plt.show()

    plt.figure()
    df_9d.plot.bar()
    plt.title('C301 {}D'.format(D[7]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.06, -0.15), ncol = 4, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_9d.png")
    plt.show()

    plt.figure()
    df_10d.plot.bar()
    plt.title('C301 {}D'.format(D[8]), fontsize=18)
    plt.ylim(0.90, 1.0)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.06, -0.15), ncol = 4, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_matrix_10d.png")
    plt.show()

def plot_JWbar(df_JW):
    plt.figure(dpi=400)
    df_JW.plot.bar()
    plt.title('C301 JW', fontsize=18)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(-0.09, -0.15), ncol = 5, borderaxespad=0, fontsize=9, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\\C301_JW.png")
    plt.show()


def main():
    df_JW = pd.read_csv("Evaluation_results\\C301_JW.csv", index_col=0)
    df_JW.columns = ['DBM_transient', 'DBM_converged', 'KPCA', 'Isomap', 'TSNE', 'UMAP']
    plot_JWbar(df_JW)
    df_matrix = pd.read_csv("Evaluation_results\\C301_matrix.csv", index_col=0)
    plot_bar(df_matrix)

if __name__ == '__main__':
    dir = 'Evaluation_results\Result_bar'
    if not os.path.exists(dir):
        os.mkdir(dir)
    main()                      # All of the result bar chart are saved in Evaluation_results