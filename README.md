# PGPS
The code and dataset of IJCAI 2023 paper "[*A Multi-Modal Neural Geometric Solver with Textual Clauses Parsed from Diagram*](https://arxiv.org/abs/2302.11097)". We propose a new neural solver **PGPSNet**, fusing multi-modal information through structural and semantic
pre-training, data augmentation, and self-limited decoding. We also construct a large-scale dataset **PGPS9K** labeled with both fine-grained diagram annotation and interpretable solution program. Our PGPSNet outperforms existing neural solvers significantly and also achieves comparable results as well-designed symbolic solvers.

<div align=center>
	<img width="400" src="images\PGDPNet.png">
</div>
<div align=center>
	Figure 1. Overview of PGPSNet solver.
</div>


## Environmental Settings
- Python version: **3.8**
- CUDA version: **10.2**
- Other settings refer to *requirements.txt*
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
For all experiments, we use **one GTX-RTX GPU** or **two TITAN Xp GPUs** for training. 

## PGPS9K Dataset
You could download the dataset from [Dataset Homepage](http://www.nlpr.ia.ac.cn/databases/CASIA-PGPS9K) and the password is '**PAL_PGPS_2023**'. 

In default, unzip the dataset file to the fold `./datasets`.

## Pre-training
As to structural and semantic pre-training, you could train the language model from scratch at [here](https://github.com/mingliangzhang2018/PGPS-Pretraining), and we also provide the pre-trained language model `LM_MODEL.pth` at [BaiduYun link](https://pan.baidu.com/s/1dVdFCVVeXDORDe5q5xpbzw) (keyword: tkbd) or [GoogleDrive link](https://drive.google.com/file/d/1h4OPMSq71aneCRWwB7muRwdsClYwXE0V/view?usp=sharing). In default, unzip the file to the fold `./`.

## Training

The default parameter configurations are set in the config file `./config/config_default.py` and the 
default training modes are displayed in `./sh_files/train.sh`, for example,

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=$((RANDOM + 10000)) \
  start.py \
  --dataset Geometry3K \
  --use_MLM_pretrain
```

You could choose dataset (**Geometry3K** / **PGPS9K**)  and whether to use the pre-training language model. The training records of the PGPSNet are saved in the folder `./log`.

## Test

The default parameter configurations are set in the config file `./config/config_default.py` and the 
default test modes are displayed in `./sh_files/test.sh`, for example,

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset Geometry3K \
--use_MLM_pretrain \
--evaluate_only \
--eval_method completion \
--resume_model log/*/best_model.pth
```
You could choose datasets (**Geometry3K** / **PGPS9K**), whether to use the pre-training language model, and evaluation methods (**completion** / **choice** / **top3**). The test records are also saved in the folder `./log` (The results of this code are 2% higher than those reported in the paper due to fine-tuning of hyperparameters). 

<div align=center>
	Table 1. Numerical answer accuracies of state-of-the-art GPS solvers (reported in the paper).
</div>
<div align=center>
	<img width="700" src="images\results.png">
</div>


## Citation

If the paper, the dataset, or the code helps you, please cite papers in the following format:
```
@inproceedings{Zhang2023PGPS,
  title     = {A Multi-Modal Neural Geometric Solver with Textual Clauses Parsed from Diagram},
  author    = {Zhang, Ming-Liang and Yin, Fei and Liu, Cheng-Lin},
  booktitle = {IJCAI},
  year      = {2023},
}

@inproceedings{Zhang2022PGDP,
  title     = {Plane Geometry Diagram Parsing},
  author    = {Zhang, Ming-Liang and Yin, Fei and Hao, Yi-Han and Liu, Cheng-Lin},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  pages     = {1636--1643},
  year      = {2022},
  month     = {7},
  doi       = {10.24963/ijcai.2022/228},
}

@article{Hao2022PGDP5KAD,
  title={PGDP5K: A Diagram Parsing Dataset for Plane Geometry Problems},
  author={Yihan Hao and Mingliang Zhang and Fei Yin and Linlin Huang},
  journal={2022 26th International Conference on Pattern Recognition (ICPR)},
  year={2022},
  pages={1763-1769}
}
```

## Acknowledge
Please let us know if you encounter any issues. You could contact with the first author (zhangmingliang2018@ia.ac.cn) or leave an issue in the github repo.
