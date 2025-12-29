<div align="center">
  <h1>Bridging Cognitive Gap: Hierarchical Description Learning for Artistic Image Aesthetics Assessment</h1> 


<h2>Motivation</h2> 

<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width: 100%" src="fig/teaser1.png">
</div>

<h2>Method</h2> 

<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width: 100%" src="fig/train_pipeline.png">
</div>
</div>




## 🔧 Installation

If you only need to infer / evaluate:

```shell
pip install -e .
```

For training, you need to further install additional dependencies as follows:

```shell
pip install -e ".[train]"
pip install flash_attn --no-build-isolation
```


## Training, Inference & Evaluation

### 🌈 Datasets

**Data Download**  
Source images are available in the [APDDv2 GitHub repository](https://github.com/BestiVictory/APDDv2). 


**Dataset Contents**  
The dataset includes the following files: 

- **Description Files**  
  - `descriptions_level_1.json`: Contains perception-level descriptions only.  
  - `descriptions_level_2.json`: Includes perception + cognition descriptions.  
  - `descriptions_level_3.json`: Provides full descriptions (perception + cognition + emotion).  

- **Scoring Data**  
  - `score.json`: Contains pre-processed score distributions. 

Arrange the folders as follows:

```
|-- ArtQuant
  |-- dataset
    |-- APDD
      |-- images
      |-- meta
        |-- descriptions_level_*.json
        |-- score.json
```
### 💪 Preprocess
The APDD dataset has been processed [here](./dataset/APDD/score.json). If you use the Score-Based Distribution Estimation algorithm (§3.2) for your dataset, you can use the [example script](preprocess/preprocess.ipynb).

### 🔥 Training

Fine-tuning needs to download the mPLUG-Owl2 weights as in [Pretrained Weights](#pretrained_weights). Put it under the [ModelZoo](./ModelZoo) folder.


- Only **2 RTX3090 GPUs** are required. 

```shell
sh train.sh
```

### 🚀 Inference

After training, you can infer with:

```shell
sh infer.sh $ONE_GPU_ID
```


### 🔍 Evaluation

After inference, you can evaluate SRCC / PLCC for quality score.

```shell
sh eval.sh
```