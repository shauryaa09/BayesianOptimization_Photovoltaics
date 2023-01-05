# BayesianOptimization_Photovoltaics
This repository contains my code that I wrote during my Master's Thesis at EPFL and CSEM. It uses Bayesian Optimization to find the optimum optical performance points of tandems and other photovoltaics.

One needs CROWM or any other optical simulator to optimize for current density and MATLAB to optimize for efficiency.

The two files data_frame_tandem.csv and PK_Si_tandem_DST_reference.txt are to opened inside the code. The file Tandem3Diode_v2.m is to be used when doing electrical simulations.

https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html is the link to the gpr function used from the sklearn library. All the arguments sent into the function are described here and that helps to decide what values should be set for those arguments.
