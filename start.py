import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import *
from core.worker import main_worker


if __name__ == '__main__':

    args = get_parser()
    cudnn.benchmark = True
    if args.debug:
        rank = 0
        local_rank = 0
    else:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank
    world_size = torch.cuda.device_count()
    args.nprocs = world_size
    dist.init_process_group(backend="nccl", init_method=args.init_method,
                            rank=local_rank, world_size=world_size)
    torch.cuda.set_device(rank % world_size)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)