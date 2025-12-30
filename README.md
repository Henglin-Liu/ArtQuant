<div align="center">
  <h1>Bridging Cognitive Gap: Hierarchical Description Learning for Artistic Image Aesthetics Assessment</h1> 
<p>
    <a href="https://arxiv.org/abs/2512.23413" target="_blank"><img src="https://img.shields.io/badge/📖-Paper-b31b1b" alt="Paper"></a>
    <a href="https://github.com/Henglin-Liu/ArtQuant" target="_blank"><img src="https://img.shields.io/badge/💻-Code-green" alt="Code"></a>
    <a href="https://huggingface.co/Liuuuu123/ArtQuant/" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue" alt="Model"></a>
    <a href="https://huggingface.co/datasets/Liuuuu123/RAD" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow" alt="Dataset"></a>
</p>

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


### 💪 Preprocess
The APDD dataset has been processed [APDD/apdd_score.json](https://huggingface.co/datasets/Liuuuu123/RAD). If you use the Score-Based Distribution Estimation algorithm (§3.2) for your dataset, you can use the [example script](preprocess/preprocess.ipynb).

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