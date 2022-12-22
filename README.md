# DBM-Transient
DBM_Transient:  A MATLAB implementation of a Deep Boltzmann Machine (DBM) in to a new transient state.
This repository provides the Matlab implementation of the DBM Transient for seizure detection on Bonn and C301 datasets. 

# Getting Started
The project is separated into three subfolders, corresponding to the two different datasets (Bonn and C301). The data set and their extracted features are stored in Data folder. The first two subfolder contain similar files, and should allow for a complete training of the model on the specific dataset. 
In addition, DBM-Transient and DBM-Converged are trained in Matlab environment. Whereas rest of the methods (KPCA, Isomap, t-SNE and UMAP) are written in Python environment. 

All the folder are structured as follows:

File train.py is the main training script, implementing the training/eval loop of the model.
File config.py contains the model's configuration settings, structured as a Python dictionary.
File prior.py contains all the prior of the adversarial autoencoder taht have been considered during the developement of the model. Some of the prior are not actually used in the final model.
File model.py contains the implementation of the layers of the AbsAE and ReL models.
File loss.py contains the implementations of all the losses used during training.
File log.py contains a collections of logging functions used to produce the final results and to check the progress of training.
File dataset.py contains the implementation of the dataset object and all data-related utilities.
