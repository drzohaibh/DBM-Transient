1. Datasets:     Bonn_data

     We divide the Bonn dataset into the following 7 cases.
"     Bonn: set_A   set_B  set_C   set_D  set_E                Each set has 100 samples, only set_E is a seizure.
"           case1:  set_A  set_E                               (non-seizure, seizure) = (100, 100)"
"           case2:  set_B  set_E                               (non-seizure, seizure) = (100, 100)"
"           case3:  set_C  set_E                               (non-seizure, seizure) = (100, 100)"
"           case4:  set_D  set_E                               (non-seizure, seizure) = (100, 100)"
"           case5:  set_A  set_C   set_D  set_E                (non-seizure, seizure) = (300, 100)"
"           case6:  set_B  set_C   set_D  set_E                (non-seizure, seizure) = (300, 100)"
"           case7:  set_A  set_B   set_C  set_D  set_E         (non-seizure, seizure) = (400, 100)"

2. Code:

      Data_preprocessing.py: 	Bonn dataset is processed by Discrete Wavelet Transform and calculated the input feature,
                              which are saved in Bonn.mat in Bonn_data folder.

                DBM_train.m: 	Using the DBM model to train Bonn data, we can get 2 to 10-dimensional outputs, respectively.
                             	Please change value c for different cases.
                             	Each layer of training is performed in RBM1--RBM2--RBM3--RBM4 ----> *.m, and you can get
                             	the result of two states['transient', 'converged']
                             	Finally, all of the results shall be saved in the folder "Different_dimension."

          Contrast_model.py:	The code in this part includes four training models: ['KPCA', 'Isomap', 'UMAP', 't-SNE']. KPCA, Isomap, and Umap
                            	can generate the dimensions from 2 to 10, and t-SNE can generate the dimensions from 2 to 3.
                             	Please change case[n] and select different cases for calculation. n∈[0, 1, 2, 3, 4, 5, 6]
					Finally, all of the data shall be saved in the folder "Different_dimension.

SVM+Fisher_Discriminiant.py: 	SVM and Fisher discriminant are used to evaluate different methods and all of the evaluation results 
					shall save in the folder "Evaluation_results". Please change the value c for different cases.

        2D_visualization.py: 	The data from different methods are used to draw 2D plots.
                             	Change case[n] and select different cases for calculation. n∈[0, 1, 2, 3, 4, 5, 6]
                           

                measure_bar: 	In order to get the different dimension performance, we plot bar charts of the detection result,
                              which shall save in "Evaluation_results/Result_bar".
