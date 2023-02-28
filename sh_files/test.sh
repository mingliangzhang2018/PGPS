CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset PGPS9K \
--use_MLM_pretrain \
--evaluate_only \
--eval_method completion

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset PGPS9K \
--use_MLM_pretrain \
--evaluate_only \
--eval_method choice

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset PGPS9K \
--use_MLM_pretrain \
--evaluate_only \
--eval_method top3 \
--beam_size 3

################################################################

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset Geometry3K \
--use_MLM_pretrain \
--evaluate_only \
--eval_method completion

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset Geometry3K \
--use_MLM_pretrain \
--evaluate_only \
--eval_method choice

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset Geometry3K \
--use_MLM_pretrain \
--evaluate_only \
--eval_method top3 \
--beam_size 3
