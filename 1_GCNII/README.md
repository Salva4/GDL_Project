# GCNII_Spektral

This repository contains a Spektral implementation of "Simple and Deep Graph Convolutional Networks" (https://arxiv.org/abs/2007.02133). It has been ported from the authors of the paper's implementation in PyG (https://github.com/chennnM/GCNII/tree/ca91f5686c4cd09cc1c6f98431a5d5b7e36acc92).

## Dependencies
- CUDA 10.1
- python 3.6.9
- spektral 1.1.0
- networkx 2.1
- scikit-learn

## Datasets

The `data` folder contains three benchmark datasets (_Cora_, _Citeseer_, _Pubmed_), and the `newdata` folder contains four datasets (_Chameleon_, _Cornell_, _Texas_, _Wisconsin_) from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn).

## Results
Testing accuracy summarized below.
| Dataset |  PyG model accuracy | Our model accuracy (Spektral) |
|:---:|:---:|:---:|
| Cora       | 85.50 | 71.79 |
| Cite       | 73.40 | 82.93 |
| Pubm       | 80.30 | 79.48 |
| Cham       | 62.48 | 20.03 |
| Corn       | 76.49 | 87.98 |
| Texa       | 77.84 | 68.85 |
| Wisc       | 81.57 | 58.57 |

## How to
Inside the folder `GCNII_Spektral`, run the python files `training_*`, (replace * for the desired dataset).




























