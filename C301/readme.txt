1. Dataset:     c301_raw_data

     c301_raw_data: raw_data for EEG signal and label for data label
     Each patient has 20 channels, including 19 EEG and 1 ECG.

     ID Seizure events (Tmin - Tmax) (s) Total Seizure time (s) Total seizure-free time (min)
     1        26       (7.4 - 84.4)             980.0                      34.5
     2        53       (2.0 - 21.8)             300.8                      45.8
     3        20       (12.1 - 45.3)            499.8                      42.5
     4        16       (1.6 - 45.3)             89.4                       49.3
     5        10       (1.6 - 8.3)              45.3                       50.1
     6        6        (5.1 - 14.1)             55.5                       49.9
     7        1        (7.8 - 7.8)              7.8                        50.7
     8        12       (3.5 - 8.6)              78.3                       49.5
     9        5        (3.1 - 18.6)             51.4                       50.0
     10       17       (1.2 - 12.5)             129.2                      48.7
     11       10       (2.3 - 11.3)             80.8                       49.5
     12       8        (2.0 - 7.2)              44.3                       50.1
     13       6        (2.7 - 9.4)              34.4                       50.2
     14       16       (2.3 - 42.1)             179.8                      47.8
     15       20       (2.3 - 13.7)             133.5                      48.6
     16       20       (2.3 - 11.7)             133.5                      48.6
     17       12       (2.0 - 15.6)             95.4                       49.2
     18       8        (1.6 - 7.4)              41.1                       50.1
     19       12       (8.6 - 22.5)             206.4                      47.4

     Sum      258                               3186.6                     912.4

     	We combine all 19 patients from the c301 datasets into"patient_1-19 by choosing the more clean dataset with 4s windows at
     	the start of seizure/non-seizure by sliding_window.m.             (non-seizure, seizure) = (9992, 757)
	The read_data.m can be used to read and analyze the C301 raw data.
	

2. Code:

      Data_preprocessing.py:	c301 datasets are processed by Discrete Wavelet Transform and calculated the input feature
                              under c301_x.mat.

                DBM_train.m: 	Please use the DBM model to train C301 data for 2 to 10-dimensional outputs.
                              Each layer of training is performed in RBM1--RBM2--RBM3--RBM4 ----> *.m, and one can get
                              the result of two states['transient', 'converged']
                              All of the results shall be saved under "Different_dimension".

          Contrast_model.py: 	The code in this part includes four training models: ['KPCA', 'Isomap', 't-SNE', 'UMAP'], 
					KPCA, Isomap, and UMAP can generate dimensions from 2 to 10, and t-SNE can generate dimensions from 2 to 3.
                              All of the results shall be saved under "Different_dimension".

SVM+Fisher_Discriminiant.py: 	SVM and Fisher discriminant are used to evaluate results from Different_dimensions, and all
                              of the results shall be saved under "Evaluation_results."

        2D_visualization.py: 	The 2D data can be used to visualize the output results for all 6 methods.
                      

                measure_bar: 	In order to evaluate the performance of different dimensions, bar plots are used, 
					which shall save the results in "Evaluation_results/Result_bar".