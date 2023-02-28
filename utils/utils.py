import os
import torch
from utils.lr_scheduler import WarmupMultiStepLR
from config import *
import datetime
import torch.distributed as dist

def save_checkpoint(state, is_best, dump_path=None):
    dump_path_recent = os.path.join(dump_path, str(state['epoch'])+'.pth')
    torch.save(state, dump_path_recent)
    if is_best:
        dump_path_best = os.path.join(dump_path,'best_model.pth')
        torch.save(state, dump_path_best)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, args, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.args = args

    def display(self, batch, lr=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if not lr is None:
            entries += ["lr: "+str(format(lr, '.6f'))]
        self.args.logger.info('\t'.join(entries))
 
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1, )):
    """
        Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_scheduler(args, optimizer):
    if args.scheduler_type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            args.scheduler_step,
            gamma=args.scheduler_factor,
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=1e-6)
    elif args.scheduler_type == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            args.scheduler_step,
            gamma=args.scheduler_factor,
            warmup_epochs=args.warm_epoch,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(args.scheduler_type))
    return scheduler

def get_optimizer(args, model):

    if args.optimizer_type == "SGD":
        params = []
        for _, p in model.named_parameters():
            if p.requires_grad:
                params.append({"params": p})
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif args.optimizer_type == "ADAM":
        params = []
        for _, p in model.named_parameters():
            if p.requires_grad:
                params.append({"params": p})
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer_type == "ADAMW":
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay
        ) 
    else:
        raise NotImplementedError("Unsupported Optimizer Type : {}".format(args.optimizer_type)) 

    return optimizer

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def set_cuda(data_dict):
    for key in data_dict:
        if torch.is_tensor(data_dict[key]):
            data_dict[key] = data_dict[key].cuda()

def initialize_logger(params, ):
    """
        Initialize the experience:
        - dump parameters
        - create a logger
    """
    while True:
        exp_id = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(os.path.join(params.dump_path, exp_id)):
            break
    params.dump_path = os.path.join(params.dump_path, exp_id)
    if params.local_rank == 0:
        os.makedirs(params.dump_path)
    # create a logger
    logger = create_logger(os.path.join(params.dump_path,'record.log'), params.local_rank)
    logger.info("============ Initialized logger ============")
    logger.info("\n"+"\n".join("\t\t\t\t%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment results will be stored in %s" % params.dump_path)
    return logger

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device) \
            .type_as(lengths) \
            .repeat(batch_size, 1) \
            .lt(lengths.unsqueeze(1))

def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r

    