# nl-fgpa

This repository contains the scripts related to the analysis presented in https://arxiv.org/abs/2305.10428 .

In partcular, we make available the piece of code related to the novel Non-Local FGPA model. The code is written in Python, and requires the numba (https://numba.pydata.org) packages.

The currently-implemented model parameters work for the specific case studied in the paper. If one wants to apply it to a different, parameters should be recalibrated. 

Set the number of threads (e.g. 4) used in numba parallelization by specifying the environmetal parameter:
export NUMBA_NUM_THREADS = 4 
