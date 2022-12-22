## DBM-Transient
DBM_Transient:  A MATLAB implementation of a Deep Boltzmann Machine (DBM) in to a new transient state.
This repository provides the Matlab implementation of the DBM Transient for seizure detection on Bonn and C301 datasets. 

## Getting Started
The project is separated into three subfolders, corresponding to the two different datasets (Bonn and C301). The data set and their extracted features are stored in Data folder. The first two subfolder contain similar files, and should allow for a complete training of the model on the specific dataset. 
In addition, DBM-Transient and DBM-Converged are trained in Matlab environment. Whereas rest of the methods (KPCA, Isomap, t-SNE and UMAP) are written in Python environment. 

Folders are structured as follows:

- File Data_preprocessing.py is to extract the features from raw data.
- File DBM_train.m is to train the DBM-transient and DBM-Coverged methods.
- File Contrast_model.py is to train the KPCA, Isomap, t-SNE and UMAP methods.
- File 2D_visualization.py is to plot 2D representation of low dimensional features.
- File SVM+Fisher_Discriminiant.py is to evaluate the methods using SVM and Fisher Discriminant function.
- File measure_bar.py is to draw the bar plots for different methods.

## Getting Help

For any other additional information, you can email at dpyang[AT]zhejianglab[DOT]com

## License

All source code from this project is released under the MIT license.
