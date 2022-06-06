# GCNII - Spektral

This repository contains a Spektral implementation of "Simple and Deep Graph Convolutional Networks" (https://arxiv.org/abs/2007.02133). It has been ported from the implementation in PyG by the authors of the paper (https://github.com/chennnM/GCNII/tree/ca91f5686c4cd09cc1c6f98431a5d5b7e36acc92).

## Dependencies
- python 3.6.9
- numpy 1.21.6
- matplotlib 3.2.2
- scipy 1.4.1
- CUDA 10.1
- tensorflow 2.8.0
- spektral 1.1.0
- networkx 2.1
- scikit-learn

## Datasets and results
--> In the report

## How to
Inside the folder `GDL_Project/1_GCNII`, run the python files `training_*`, (replace * for the desired dataset).

## Results
In the `results/Results.ipynb` file, there are the results on which the report is based. 

## Output
Inside the folder `output`, there is plotted the evolution of the validation accuracy and the goal accuracy (the one achieved by the original PyG implementation), for each dataset.




























