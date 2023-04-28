# PGPS
The code and dataset of IJCAI 2023 paper "A Multi-Modal Neural Geometric Solver with Textual Clauses Parsed from Diagram".

## PGPS9K Dataset
You could download the dataset from [Dataset Homepage](http://www.nlpr.ia.ac.cn/databases/CASIA-PGPS9K).

#### Format of Annotation
```
"prob_id": {  
    "diagram": ..., # name of diagram 
    "text": ..., # content of textual problem
    "parsing_stru_seqs": ..., # structural clauses
    "parsing_sem_seqs": ..., # semantic clauses
    "expression": ..., # solution program
    "answer": ..., # numerical answer
    "choices": ..., # four numerical candidates
    "type": ..., # knowledge type of question
    "book": ..., # textbook name 
    "page": ..., # page location 
}
```