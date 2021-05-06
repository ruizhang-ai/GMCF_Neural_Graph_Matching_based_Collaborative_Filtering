# Neural Graph Matching based Collaborative Filtering (GMCF)

This is our implementation for the paper:

Yixin Su, Rui Zhang*, Sarah Erfani, Junhao Gan, **Neural Graph Matching based Collaborative Filtering**. *International Conference on Research and Development in Information Retrieval* (SIGIR) 2021.

## Description

User and item attributes are essential side-information; their interactions (i.e., their co-occurrence in the sample data) can significantly enhance prediction accuracy in various recommender systems. We identify two different types of attribute interactions, *inner interactions* and *cross interactions*: inner interactions are those between *only* user attributes or those between *only* item attributes; cross interactions are those between user attributes and item attributes. Existing models do not distinguish the two types of attribute interactions, which may not be the most effective way to exploit the information carried by the interactions. 

<p align="center">
  <img src="https://github.com/suyixin12123/GMCF/blob/main/img/running_exmaple.png", alt="Differences" width="600">
  <p align="center"><em>Figure1: Illustration of the differences between our GMCF model (left) and existing graph-based work (right). GMCF treats attribute interactions differently in a structure of graph matching, while existing work treats all attribute interactions equally.</em></p>
</p>


To address this drawback, we propose a neural Graph Matching based Collaborative Filtering model (GMCF), which effectively captures the two types of attribute interactions through modeling and aggregating attribute interactions in a graph matching structure for recommendation. In our model, the two essential recommendation procedures, characteristic learning and preference matching, are explicitly conducted through graph learning (based on inner interactions) and node matching (based on cross interactions), respectively.

<p align="center">
  <img src="https://github.com/suyixin12123/GMCF/blob/main/img/GMCF_structure.png", alt="Model Structure" width="800">
  <p align="center"><em>Figure2: An Overview of the GMCF Model.</em></p>
</p>



## What are in this Repository
This repository contains the following contents:

```
/
├── code/                   --> (The folder containing the source code)
|   ├── dataloader.py       --> (The code to proceed the data into code-usable format)
|   ├── main.py             --> (The main code file. The code is run through this file)
|   ├── model.py            --> (Contains the function of our GMCF model.)
|   ├── train.py            --> (Contains the code to train and evaluate our GMCF model.)
├── data/                   --> (The folder containing three used datasets)   
|   ├── book-crossing/      --> (The book-crossing dataset.)
|   ├── ml-1m/              --> (The MovieLens 1M dataset.)
|   ├── taobao/             --> (The Taobao dataset.)
├── img/                    --> (The images for README (not used for the code))   
|   ├── GMCF_structure.png  --> (The overall structure of our GMCF model)
|   ├── running_exmaple.png --> (The differences between GMCF and existing model)
├── LICENCE                 --> (The licence file)
```

## Run our code

To run our code, please follow the instructions in our [code/](code/) folder.

## Cite our paper

Please credit our work by citing the following paper:

```
@inproceedings{su2021neural,
  title={Neural Graph Matching based Collaborative Filtering},
  author={Su, Yixin and Zhang, Rui and Erfani, Sarah and Junhao Gan},
  booktitle={Proceedings of the 44th International Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2021}
}
```

