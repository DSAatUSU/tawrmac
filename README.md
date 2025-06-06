# TAWRMAC

In this paper, we propose a novel dynamic
graph representation learning framework featuring Temporal
Anonymous Walks with Restart, Memory Augmentation and Neighbor Co-occurrence

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

#### Best Hyperparameters for Our Model

Table 1: Configuration of TAWRMAC Across All Datasets
| **Module** | **Hyper-parameter**                                       | **Value** |
| ---------- | --------------------------------------------------------- | --------- |
| **All**    | Dropout                                                   | 0.1       |
|            | Learnable time encoding dimension (\$d\_{\phi\_1}\$)      | 100       |
| **MAE**    | Fixed time encoding dimension (\$d\_{\phi\_2}\$)          | 20        |
|            | Memory dimension (\$d\_m\$)                               | 172       |
|            | Number of sampled neighbors for graph convolution (\$k\$) | 10        |
|            | Number of graph attention heads                           | 2         |
|            | Number of graph convolution layers (\$L\$)                | 1         |
| **NCE**    | Neighbor co-occurrence encoding dimension (\$d\_{ce}\$)   | 10        |
| **TAWR**   | Number of attention heads for walk encoding               | 4         |
|            | Length of each walk (\$w\$)                               | 4         |
|            | Time scaling factor (\$\alpha\$)                          | 1e-6      |


Table 2: Dataset-Specific Parameters
| **Dataset**         | **M**       | **r**         | **d\_v**       | **d\_w**       | **NbSS** |
| ------------------- | ----------- | ------------- | -------------- | -------------- | -------- |
| Wikipedia\*         | 10 / 10 / 1 | 32 / 32 / 128 | 100 / 100 / 10 | 172 / 172 / 10 | recent   |
| Reddit              | 10          | 32            | 100            | 172            | recent   |
| MOOC                | 10          | 32            | 100            | 172            | recent   |
| LastFM\*            | 10          | 32 / 32 / 4   | 100            | 172            | recent   |
| Enron\*             | 10 / 1 / 1  | 32 / 8 / 8    | 100 / 10 / 10  | 172 / 10 / 10  | recent   |
| UCI\*               | 10          | 32 / 4 / 4    | 100            | 172            | recent   |
| Flights             | 20          | 32            | 100            | 172            | recent   |
| Canadian Parliament | 30          | 500           | 100            | 172            | uniform  |
| US Legislature\*    | 10          | 200 / 32 / 32 | 100            | 172            | recent   |
| UN Trade            | 20          | 200           | 100            | 172            | uniform  |
| UN Vote             | 20          | 100           | 100            | 172            | uniform  |
| Contact             | 10          | 32            | 100            | 172            | recent   |


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
