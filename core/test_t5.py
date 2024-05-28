import time
from utils import *

# def validate(args, val_loader, model, tgt_lang):
#     batch_time = AverageMeter('Time', ':5.3f')
#     acc_ans = AverageMeter('Ans_Acc', ':5.4f')
#     acc_eq = AverageMeter('Eq_Acc', ':5.4f')
#     progress = ProgressMeter(len(val_loader), [batch_time, acc_ans, acc_eq], args, prefix='Test: ')
    
#     # switch to evaluate mode
#     model.eval()

#     with torch.no_grad():
#         end = time.time()
#         for i, batch in enumerate(val_loader):
#             inputs = {k: v.cuda() for k, v in batch.items() if k != 'labels'}
#             labels = batch['labels'].cuda()
            
#             # compute output
#             outputs = model(**inputs)
#             print(outputs)
#             args.logger.info(outputs)

#             if args.eval_method == "completion":
#                 acc1, acc2 = compute_exp_result_comp(outputs, inputs, labels, tgt_lang)
#             elif args.eval_method == "choice":
#                 acc1, acc2 = compute_exp_result_choice(outputs, inputs, labels, tgt_lang)
#             elif args.eval_method == "top3":
#                 acc1, acc2 = compute_exp_result_topk(outputs, inputs, labels, tgt_lang, k_num=3)

#             torch.distributed.barrier()

#             reduced_acc_ans = reduce_mean(torch.tensor([acc1]).cuda(), args.nprocs)
#             reduced_acc_eq = reduce_mean(torch.tensor([acc2]).cuda(), args.nprocs)

#             acc_ans.update(reduced_acc_ans.item(), len(labels))
#             acc_eq.update(reduced_acc_eq.item(), len(labels))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % args.print_freq == 0:
#                 progress.display(i)

#     return acc_ans.avg, acc_eq.avg

import time
import torch
from transformers import T5Tokenizer
from utils import AverageMeter, ProgressMeter

def validate(args, val_loader, model, tgt_lang):
    batch_time = AverageMeter('Time', ':5.3f')
    acc_ans = AverageMeter('Ans_Acc', ':5.4f')
    acc_eq = AverageMeter('Eq_Acc', ':5.4f')
    progress = ProgressMeter(len(val_loader), [batch_time, acc_ans, acc_eq], args, prefix='Test: ')
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            inputs = {k: v.cuda() for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].cuda()
            
            # Create decoder_input_ids by shifting the labels to the right
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
            
            # compute output
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
            test_preds = outputs.logits.argmax(dim=-1)  # Assuming beam_size x token_list
            print(test_preds)
            args.logger.info(test_preds)


            if args.eval_method == "completion":
                acc1, acc2 = compute_exp_result_comp(outputs, inputs, labels, tgt_lang)
            elif args.eval_method == "choice":
                acc1, acc2 = compute_exp_result_choice(outputs, inputs, labels, tgt_lang)
            elif args.eval_method == "top3":
                acc1, acc2 = compute_exp_result_topk(outputs, inputs, labels, tgt_lang, k_num=3)

            torch.distributed.barrier()

            reduced_acc_ans = reduce_mean(torch.tensor([acc1]).cuda(), args.nprocs)
            reduced_acc_eq = reduce_mean(torch.tensor([acc2]).cuda(), args.nprocs)

            acc_ans.update(reduced_acc_ans.item(), len(labels))
            acc_eq.update(reduced_acc_eq.item(), len(labels))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return acc_ans.avg, acc_eq.avg
