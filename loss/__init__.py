from .loss import *
from config import criterion_list


def get_criterion(args):   
    # create model
    if args.criterion in criterion_list:
        return eval(args.criterion)(args)
    else:
        raise NotImplementedError("Unsupported Loss Criterion : {}".format(args.criterion)) 