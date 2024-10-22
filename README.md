# ROSE
PyTorch implementation for "ROSE: Robust Cross Supervision with Neighborhood Mining for Source-free Graph Domain Adaptation".

## Abstract
Graph neural networks have achieved impressive performance in graph domain adaptation, but they typically require access to source graphs, which could be infeasible due to data privacy in real-world applications. To address this problem, we study an underexplored yet realistic problem of source-free graph domain adaptation, which transfers knowledge from source models instead of source graphs to a target domain. We propose a new approach named Robust Cross Supervision with Neighborhood Mining (ROSE) for this problem.ROSE consists of a message passing branch and a graph kernel branch, which infer graph topological information from complementary views. To promote more reliable optimization of pseudo-labeling, we explore the neighborhood for each graph sample in the embedding space to divide them into informative samples and basic samples, which are incorporated into a meta-learning optimization framework for robustness. To incorporate both branches into a unified and robust optimization framework, we use the prediction of one branch to supervise the other branch in an alternative manner. Extensive experiments on benchmark datasets demonstrate the effectiveness of the proposed ROSE compared with a range of baselines. 


## Requirements

* python==3.7.10
* torch==1.8.0+cu102
* torch-geometric==1.7.2

To install `torch-geometric==1.7.2` for `torch==1.8.0+cu102`
```txt
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_sparse-0.6.10-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==1.7.2
```
 