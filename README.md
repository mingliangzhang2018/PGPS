# PGPS
The code and dataset of IJCAI 2023 paper "[*A Multi-Modal Neural Geometric Solver with Textual Clauses Parsed from Diagram*](https://arxiv.org/abs/2302.11097)". We propose a new neural solver **PGPSNet**, fusing multi-modal information through structural and semantic
pre-training, data augmentation, and self-limited decoding. We also construct a large-scale dataset **PGPS9K** labeled with both fine-grained diagram annotation and interpretable solution program. Our PGPSNet outperforms existing neural solvers significantly and also achieves comparable results as well-designed symbolic solvers.

<div align=center>
	<img width="400" src="images\PGDPNet.png">
</div>
<div align=center>
	Figure 1. Overview of PGPSNet solver.
</div>

<div align=center>
	<img width="800" src="images\Pre-training.png">
</div>
<div align=center>
	Figure 2. Pipeline of structural and semantic pre-training.
</div>

## PGPS9K Dataset
You could download the dataset from [Dataset Homepage](http://www.nlpr.ia.ac.cn/databases/CASIA-PGPS9K).

<div align=center>
	<img width="750" src="images\datasets.png">
</div>
<div align=center>
	Figure 3. Example presentation of PGPS9K dataset.
</div>

#### Format of Solution Program
<div align=center>
	<img width="400" src="images\Annotation_Sample.png">
</div>
<div align=center>
	Figure 4. Annotation of solution program and its interpretability.
</div>

#### Format of Annotation File
```
"prob_id": {  
    "diagram": ...,    # name of diagram 
    "text": ...,    # content of textual problem
    "parsing_stru_seqs": ...,    # structural clauses
    "parsing_sem_seqs": ...,    # semantic clauses
    "expression": ...,    # solution program
    "answer": ...,    # numerical answer
    "choices": ...,    # four numerical candidates
    "type": ...,    # knowledge type of question
    "book": ...,    # textbook name 
    "page": ...,    # page location 
}
```



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