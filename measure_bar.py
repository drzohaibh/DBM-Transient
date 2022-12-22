import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_bar(df_matrix, c):

    Method = ['transient', 'converged', 'KPCA', 'Isomap', 'TSNE', 'UMAP']
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
        df_2d[M] = df_matrix[c + ' ' + M + ' 2D']
        df_3d[M] = df_matrix[c + ' ' + M + ' 3D']
        if M != 'TSNE':
            df_4d[M] = df_matrix[c + ' ' + M + ' 4D']
            df_5d[M] = df_matrix[c + ' ' + M + ' 5D']
            df_6d[M] = df_matrix[c + ' ' + M + ' 6D']
            df_7d[M] = df_matrix[c + ' ' + M + ' 7D']
            df_8d[M] = df_matrix[c + ' ' + M + ' 8D']
            df_9d[M] = df_matrix[c + ' ' + M + ' 9D']
            df_10d[M] = df_matrix[c + ' ' + M + ' 10D']
    plt.figure()
    df_2d.plot.bar()
    plt.title('{} {}D'.format(c, D[0]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_2d.png".format(c))
    plt.show()

    plt.figure()
    df_3d.plot.bar()
    plt.title('{} {}D'.format(c, D[1]).format(D[1]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_3d.png".format(c))
    plt.show()

    plt.figure()
    df_4d.plot.bar()
    plt.title('{} {}D'.format(c, D[2]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_4d.png".format(c))
    plt.show()

    plt.figure()
    df_5d.plot.bar()
    plt.title('{} {}D'.format(c, D[3]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_5d.png".format(c))
    plt.show()

    plt.figure()
    df_6d.plot.bar()
    plt.title('{} {}D'.format(c, D[4]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_6d.png".format(c))
    plt.show()

    plt.figure()
    df_7d.plot.bar()
    plt.title('{} {}D'.format(c, D[5]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_7d.png".format(c))
    plt.show()

    plt.figure()
    df_8d.plot.bar()
    plt.title('{} {}D'.format(c, D[6]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_8d.png".format(c))
    plt.show()

    plt.figure()
    df_9d.plot.bar()
    plt.title('{} {}D'.format(c, D[7]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False) # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_9d.png".format(c))
    plt.show()

    plt.figure()
    df_10d.plot.bar()
    plt.title('{} {}D'.format(c, D[8]), fontsize=18)
    plt.ylim(0.60, 1.0)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_matrix_10d.png".format(c))
    plt.show()

def plot_JWbar(df_JW, c):
    plt.figure(figsize=(8, 4), dpi=400)
    df_JW.plot.bar()
    plt.title('{} JW'.format(c), fontsize=18)
    legend = plt.legend(loc=4, borderaxespad=0, fontsize=18, frameon=False)  # , frameon=False
    frame = legend.get_frame()
    frame.set_facecolor('none')
    plt.savefig("Evaluation_results\Result_bar\{}_JW.png".format(c))
    plt.show()

def main(c):
    df_JW = pd.read_csv("Evaluation_results\Bonn_{}_JW.csv".format(c), index_col=0)
    df_JW.columns = ['DBM_transient', 'DBM_converged', 'KPCA', 'Isomap', 'TSNE', 'UMAP']
    plot_JWbar(df_JW, c)
    df_matrix = pd.read_csv("Evaluation_results\Bonn_{}_matrix.csv".format(c), index_col=0)
    plot_bar(df_matrix, c)


if __name__ == '__main__':
    dir = 'Evaluation_results\Result_bar'
    if not os.path.exists(dir):
        os.mkdir(dir)

    case = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7']
    c = case[0]                                                               # change the case
    main(c)                      # All of the result bar chart are saved in Evaluation_results