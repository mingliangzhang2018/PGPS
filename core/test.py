import time
from utils import *

def validate(args, val_loader, model, tgt_lang):
    
    batch_time = AverageMeter('Time', ':5.3f')
    acc_ans = AverageMeter('Ans_Acc', ':5.4f')
    acc_eq = AverageMeter('Eq_Acc', ':5.4f')
    progress = ProgressMeter(len(val_loader), [batch_time, acc_ans, acc_eq], args, prefix='Test: ')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (diagrams, text_dict, var_dict, exp_dict) in enumerate(val_loader):
            # set cuda for input data
            diagrams = diagrams.cuda()
            set_cuda(text_dict), set_cuda(var_dict), set_cuda(exp_dict)
            # compute output
            output = model(diagrams, text_dict, var_dict, exp_dict, is_train=False)
            if args.eval_method == "completion":
                acc1, acc2 = compute_exp_result_comp(output, var_dict, exp_dict, tgt_lang)
            elif args.eval_method == "choice":
                acc1, acc2 = compute_exp_result_choice(output, var_dict, exp_dict, tgt_lang)
            elif args.eval_method == "top3":
                acc1, acc2 = compute_exp_result_topk(output, var_dict, exp_dict, tgt_lang, k_num=3)

            torch.distributed.barrier()
            
            reduced_acc_ans = reduce_mean(torch.tensor([acc1]).cuda(), args.nprocs)
            reduced_acc_eq = reduce_mean(torch.tensor([acc2]).cuda(), args.nprocs)

            acc_ans.update(reduced_acc_ans.item(), len(diagrams))
            acc_eq.update(reduced_acc_eq.item(), len(diagrams))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return acc_ans.avg, acc_eq.avg
