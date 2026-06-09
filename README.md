# TAWRMAC

In this paper, we propose a novel dynamic graph representation learning framework featuring Temporal Anonymous Walks with Restart, Memory Augmentation and Neighbor Co-occurrence.

## Running the experiments

### Requirements

Install packages in ```requirements.txt``` (with python >= 3.9):

### Dataset and Preprocessing

#### Download the public data
The datasets come from [Towards Better Evaluation for Dynamic Link Prediction](https://openreview.net/forum?id=1GVpwr2Tfdg), 
which can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o). 
Please download them, and store their csv files in a folder named
```data/```.

#### Preprocess the data
We use the `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
For example:
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
```


### Model Training

Dynamic link prediction task:
```{bash}
python train_link_prediction.py --negative_sampling_strategy random --use_memory --enable_walk --enable_dynamic_restart --enable_neighbor_cooc --n_runs 5
```

Dynamic node classification (this requires a trained model from 
the link prediction task):
```{bash}
python train_node_classification.py --use_memory --enable_walk --enable_dynamic_restart --enable_neighbor_cooc --n_runs 5 --model_date 20241206102940
```

## Citation

This repository accompanies our paper accepted at **The Web Conference (TheWebConf) 2026**.

Please cite the following paper:
```
@inproceedings{farokhi2026tawrmac,
author = {Farokhi, Soheila and Qi, Xiaojun and Karimi, Hamid},
title = {TAWRMAC: A Novel Dynamic Graph Representation Learning Method},
year = {2026},
isbn = {9798400723070},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3774904.3792163},
doi = {10.1145/3774904.3792163},
booktitle = {Proceedings of the ACM Web Conference 2026},
pages = {684–695},
numpages = {12},
keywords = {dynamic graph representation learning, graph neural network, temporal walks, link prediction, node classification},
location = {United Arab Emirates},
series = {WWW '26}
}
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
