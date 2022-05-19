# PDN - Spektral

This repository contains a Spektral implementation of "Pathfinder Discovery Networks for Neural Message Passing" (https://arxiv.org/abs/2010.12878). It has been ported from the implementation in PyG by the authors of the paper (https://github.com/benedekrozemberczki/PDN).

## Dependencies
- python 3.6.9
- numpy 1.21.6
- matplotlib 3.2.2
- CUDA 10.1
- tensorflow 2.8.0
- spektral 1.1.0

## Datasets

The `data` folder contains the data for two graphs; one for training and the other one for testing, which are two disjoint subgraphs of the original one, given in the github referred above (our model can't handle bigger datasets). Therefore, the results with the original PyG implementation are not comparable. 

## Results
Our implementation achieves, for the aforementioned subgraphs of the one originally given, a 71.88% of test accuracy (over 4 classes). 

## How to
Inside the folder `GDL_Project/2_PDN`, run the python files `main.py`.

## Output
Inside the folder `output`, there is plotted the evolution of the training accuracy.


























