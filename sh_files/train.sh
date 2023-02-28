CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset Geometry3K \
--use_MLM_pretrain

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM + 10000)) \
start.py \
--dataset PGPS9K \
--use_MLM_pretrain