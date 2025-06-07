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

### Best Hyperparameters for Our Model

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


Table 2: Configuration of number of temporal walks with restart (**M**), number of sampled neighbors in NCE (**r**), positional frequency vector dimension (**\$d\_v\$**), walk encoding dimension (**\$d\_w\$**), and neighbor sampling strategies (**NbSS**) across different datasets.
| **Dataset**         | **\$M\$**       | **\$r\$**         | **\$d\_v\$**       | **\$d\_w\$**       | **NbSS** |
| ------------------- | ----------- | ------------- | -------------- | -------------- | -------- |
| Wikipedia\*         | 10 / 10 / 1 | 32 / 32 / 128 | 100 / 100 / 10 | 172 / 172 / 10 | recent   |
| Reddit              | 10          | 32            | 100            | 172            | recent   |
| MOOC                | 10          | 32            | 100            | 172            | recent   |
| LastFM\*            | 10          | 32 / 32 / 4   | 100            | 172            | recent   |
| Enron\*             | 10 / 1 / 1  | 32 / 8 / 8    | 100 / 10 / 10  | 172 / 10 / 10  | recent   |
| UCI\*               | 10          | 32 / 4 / 4    | 100            | 172            | recent   |
| Flights             | 20          | 32            | 100            | 172            | recent   |
| Can. Parl. | 30          | 500           | 100            | 172            | uniform  |
| US Legis.\*    | 10          | 200 / 32 / 32 | 100            | 172            | recent   |
| UN Trade            | 20          | 200           | 100            | 172            | uniform  |
| UN Vote             | 20          | 100           | 100            | 172            | uniform  |
| Contact             | 10          | 32            | 100            | 172            | recent   |

*For columns with multiple values, the first value corresponds to random NS, the second to historical NS, and the third to inductive NS.





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



### AP Results on Inductive Dynamic Link Prediction


Table 3: Random Negative Sampling Strategy

| Dataset     | JODIE           | DyRep           | TGAT            | TGN             | CAWN            | TCL             | GraphMixer      | DyGFormer        | TAWRMAC          |
|-------------|-----------------|-----------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Wikipedia   | 94.82 ± 0.20    | 92.43 ± 0.37    | 96.22 ± 0.07     | 97.83 ± 0.04     | 98.24 ± 0.03     | 96.22 ± 0.17     | 96.65 ± 0.02     | _98.59 ± 0.03_   | **98.93 ± 0.03** |
| Reddit      | 96.50 ± 0.13    | 96.09 ± 0.11    | 97.09 ± 0.04     | 97.50 ± 0.07     | 98.62 ± 0.01     | 94.09 ± 0.07     | 95.26 ± 0.02     | _98.84 ± 0.02_   | **98.99 ± 0.05** |
| MOOC        | 79.63 ± 1.92    | 81.07 ± 0.44    | 85.50 ± 0.19     | _89.04 ± 1.17_   | 81.42 ± 0.24     | 80.60 ± 0.22     | 81.41 ± 0.21     | 86.96 ± 0.43     | **91.14 ± 0.82** |
| LastFM      | 81.61 ± 3.82    | 83.02 ± 1.48    | 78.63 ± 0.31     | 81.45 ± 4.29     | 89.42 ± 0.07     | 73.53 ± 1.66     | 82.11 ± 0.42     | **94.23 ± 0.09** | _93.07 ± 1.37_   |
| Enron       | 80.72 ± 1.39    | 74.55 ± 3.95    | 67.05 ± 1.51     | 77.94 ± 1.02     | 86.35 ± 0.51     | 76.14 ± 0.79     | 75.88 ± 0.48     | **89.76 ± 0.34** | _89.45 ± 0.12_   |
| UCI         | 79.86 ± 1.48    | 57.48 ± 1.87    | 79.54 ± 0.48     | 88.12 ± 2.05     | 92.73 ± 0.06     | 87.36 ± 2.03     | 91.19 ± 0.42     | _94.54 ± 0.12_   | **95.08 ± 0.33** |
| Flights     | 94.74 ± 0.37    | 92.88 ± 0.73    | 88.73 ± 0.33     | 95.03 ± 0.60     | 97.06 ± 0.02     | 83.41 ± 0.07     | 83.03 ± 0.05     | **97.79 ± 0.02** | _97.37 ± 0.12_   |
| Can. Parl.  | 53.92 ± 0.94    | 54.02 ± 0.76    | 55.18 ± 0.79     | 54.10 ± 0.93     | 55.80 ± 0.69     | 54.30 ± 0.66     | _55.91 ± 0.82_   | **87.74 ± 0.71** | 55.90 ± 0.83     |
| US Legis.   | 54.93 ± 2.29    | 57.28 ± 0.71    | 51.00 ± 3.11     | _58.63 ± 0.37_   | 53.17 ± 1.20     | 52.59 ± 0.97     | 50.71 ± 0.76     | 54.28 ± 2.87     | **59.28 ± 1.12** |
| UN Trade    | 59.65 ± 0.77    | 57.02 ± 0.69    | 61.03 ± 0.18     | 58.31 ± 3.15     | **65.24 ± 0.21** | 62.21 ± 0.12     | 62.17 ± 0.31     | _64.55 ± 0.62_   | 63.86 ± 1.83     |
| UN Vote     | 56.64 ± 0.96    | 54.62 ± 2.22    | 52.24 ± 1.46     | **58.85 ± 2.51** | 49.94 ± 0.45     | 51.60 ± 0.97     | 50.68 ± 0.44     | 55.93 ± 0.39     | _58.16 ± 1.63_   |
| Contact     | 94.34 ± 1.45    | 92.18 ± 0.41    | 95.87 ± 0.11     | 93.82 ± 0.99     | 89.55 ± 0.30     | 91.11 ± 0.12     | 90.59 ± 0.05     | _98.03 ± 0.02_   | **98.23 ± 0.06** |


Table 4: Historical Negative Sampling Strategy

| Dataset     | JODIE           | DyRep           | TGAT            | TGN             | CAWN            | TCL             | GraphMixer      | DyGFormer        | TAWRMAC          |
|-------------|-----------------|-----------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Wikipedia   | 68.69 ± 0.39    | 62.18 ± 1.27    | 84.17 ± 0.22     | 81.76 ± 0.32     | 67.27 ± 1.63     | 82.20 ± 2.18     | **87.60 ± 0.30** | 71.42 ± 4.43     | _84.37 ± 0.25_   |
| Reddit      | 62.34 ± 0.54    | 61.60 ± 0.72    | 63.47 ± 0.36     | 64.85 ± 0.85     | 63.67 ± 0.41     | 60.83 ± 0.25     | 64.50 ± 0.26     | _65.37 ± 0.68_   | **67.95 ± 0.65** |
| MOOC        | 63.22 ± 1.55    | 62.93 ± 1.24    | 76.73 ± 0.29     | 77.07 ± 3.41     | 74.68 ± 0.68     | 74.27 ± 0.53     | 74.00 ± 0.97     | **80.82 ± 0.30** | _79.78 ± 4.67_   |
| LastFM      | 70.39 ± 4.31    | 71.45 ± 1.76    | 76.27 ± 0.25     | 66.65 ± 6.11     | 71.33 ± 0.47     | 65.78 ± 0.65     | _76.42 ± 0.22_   | 76.35 ± 0.52     | **76.53 ± 0.23** |
| Enron       | 65.86 ± 3.71    | 62.08 ± 2.27    | 61.40 ± 1.31     | 62.91 ± 1.16     | 60.70 ± 0.36     | 67.11 ± 0.62     | _72.37 ± 1.37_   | 67.07 ± 0.62     | **77.25 ± 0.77** |
| UCI         | 63.11 ± 2.27    | 52.47 ± 2.06    | 70.52 ± 0.93     | 70.78 ± 0.78     | 64.54 ± 0.47     | 76.71 ± 1.00     | _81.66 ± 0.49_   | 72.13 ± 1.87     | **83.48 ± 0.15** |
| Flights     | 61.01 ± 1.65    | 62.83 ± 1.31    | _64.72 ± 0.36_   | 59.31 ± 1.43     | 56.82 ± 0.57     | 64.50 ± 0.25     | **65.28 ± 0.24** | 57.11 ± 0.21     | 55.77 ± 0.33     |
| Can. Parl.  | 52.60 ± 0.88    | 52.28 ± 0.31    | 56.72 ± 0.47     | 54.42 ± 0.77     | 57.14 ± 0.07     | 55.71 ± 0.74     | 55.84 ± 0.73     | **87.40 ± 0.85** | _57.76 ± 0.33_   |
| US Legis.   | 52.94 ± 2.11    | **62.10 ± 1.41**| 51.83 ± 3.95     | _61.18 ± 1.10_   | 55.56 ± 1.71     | 53.87 ± 1.41     | 52.03 ± 1.02     | 56.31 ± 3.46     | 58.41 ± 2.18     |
| UN Trade    | 55.46 ± 1.19    | 55.49 ± 0.84    | 55.28 ± 0.71     | 52.80 ± 3.19     | 55.00 ± 0.38     | _55.76 ± 1.03_   | 54.94 ± 0.97     | 53.20 ± 1.07     | **56.51 ± 3.13** |
| UN Vote     | 61.04 ± 1.30    | 60.22 ± 1.78    | 53.05 ± 3.10     | _63.74 ± 3.00_   | 47.98 ± 0.84     | 54.19 ± 2.17     | 48.09 ± 0.43     | 52.63 ± 1.26     | **65.54 ± 0.85** |
| Contact     | 90.42 ± 2.34    | 89.22 ± 0.66    | **94.15 ± 0.45** | 88.13 ± 1.50     | 74.20 ± 0.80     | 90.44 ± 0.17     | 89.91 ± 0.36     | _93.56 ± 0.52_   | 93.32 ± 0.10     |


Table 5: Inductive Negative Sampling Strategy

| Dataset     | JODIE           | DyRep           | TGAT            | TGN             | CAWN            | TCL             | GraphMixer      | DyGFormer        | TAWRMAC          |
|-------------|-----------------|-----------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Wikipedia   | 68.70 ± 0.39    | 62.19 ± 1.28    | 84.17 ± 0.22     | 81.77 ± 0.32     | 67.24 ± 1.63     | 82.20 ± 2.18     | **87.60 ± 0.29** | 71.42 ± 4.43     | _86.12 ± 1.24_   |
| Reddit      | 62.32 ± 0.54    | 61.58 ± 0.72    | 63.40 ± 0.36     | 64.84 ± 0.84     | 63.65 ± 0.41     | 60.81 ± 0.26     | 64.49 ± 0.25     | _65.35 ± 0.82_   | **65.49 ± 0.77** |
| MOOC        | 63.22 ± 1.55    | 62.92 ± 1.24    | 76.72 ± 0.30     | 77.07 ± 3.40     | 74.69 ± 0.68     | 74.28 ± 0.53     | 73.99 ± 0.97     | _80.82 ± 0.30_   | **81.43 ± 2.85** |
| LastFM      | 70.39 ± 4.31    | 71.45 ± 1.75    | 76.28 ± 0.25     | 69.46 ± 4.65     | 71.33 ± 0.47     | 65.78 ± 0.65     | _76.42 ± 0.22_   | 76.35 ± 0.52     | **77.47 ± 0.26** |
| Enron       | 65.86 ± 3.71    | 62.08 ± 2.27    | 61.40 ± 1.30     | 62.90 ± 1.16     | 60.72 ± 0.36     | 67.11 ± 0.62     | _72.37 ± 1.38_   | 67.07 ± 0.62     | **77.28 ± 0.59** |
| UCI         | 63.16 ± 2.27    | 52.47 ± 2.09    | 70.49 ± 0.93     | 70.73 ± 0.79     | 64.54 ± 0.47     | 76.65 ± 0.99     | _81.64 ± 0.49_   | 72.13 ± 1.86     | **82.85 ± 0.65** |
| Flights     | 61.01 ± 1.66    | 62.83 ± 1.31    | _64.72 ± 0.37_   | 59.32 ± 1.45     | 56.82 ± 0.56     | 64.50 ± 0.25     | **65.29 ± 0.24** | 57.11 ± 0.20     | 55.38 ± 0.82     |
| Can. Parl.  | 52.58 ± 0.86    | 52.24 ± 0.28    | 56.46 ± 0.50     | 54.18 ± 0.73     | 57.06 ± 0.08     | 55.46 ± 0.69     | 55.76 ± 0.65     | **87.22 ± 0.82** | _58.32 ± 0.90_   |
| US Legis.   | 52.94 ± 2.11    | **62.10 ± 1.41**| 51.83 ± 3.95     | _61.18 ± 1.10_   | 55.56 ± 1.71     | 53.87 ± 1.41     | 52.03 ± 1.02     | 56.31 ± 3.46     | 58.87 ± 2.77     |
| UN Trade    | 55.43 ± 1.20    | 55.42 ± 0.87    | 55.58 ± 0.68     | 52.80 ± 3.24     | 54.97 ± 0.38     | _55.66 ± 0.98_   | 54.88 ± 1.01     | 52.56 ± 1.70     | **55.98 ± 2.15** |
| UN Vote     | 61.17 ± 1.33    | 60.29 ± 1.79    | 53.08 ± 3.10     | _63.71 ± 2.97_   | 48.01 ± 0.82     | 54.13 ± 2.16     | 48.10 ± 0.40     | 52.61 ± 1.25     | **66.71 ± 2.15** |
| Contact     | 90.43 ± 2.33    | 89.22 ± 0.65    | **94.14 ± 0.45** | 88.12 ± 1.50     | 74.19 ± 0.81     | 90.43 ± 0.17     | 89.91 ± 0.36     | _93.55 ± 0.52_   | 93.71 ± 0.45     |
