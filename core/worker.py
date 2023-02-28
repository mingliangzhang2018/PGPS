import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
from core.train import *
from core.test import *
from utils import *
from core.network import get_model
from loss import get_criterion
from datasets import get_dataloader


def main_worker(args):

    args.logger = initialize_logger(args)
    train_loader, train_sampler, val_loader, src_lang, tgt_lang = get_dataloader(args)
    model = get_model(args, src_lang, tgt_lang).cuda()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    criterion = get_criterion(args)
    start_epoch = 0
    
    # resume model
    if not args.resume_model =='':
        resume_model_dict = model.load_model(args.resume_model)
        optimizer.load_state_dict(resume_model_dict['optimizer'])
        scheduler.load_state_dict(resume_model_dict['scheduler'])
        start_epoch = resume_model_dict["epoch"]+1
        args.logger.info("The whole model has been loaded from "+ args.resume_model)
        args.logger.info("The model resumes from epoch "+ str(resume_model_dict["epoch"]))
        if args.evaluate_only:
            acc_ans, acc_eq = validate(args, val_loader, model, tgt_lang)
            args.logger.info("----------Epoch:{:>3d}, test answer_acc {:>5.4f}, equation_acc {:>5.4f} ---------" \
                                            .format(resume_model_dict["epoch"], acc_ans, acc_eq))
            return
    else:
        args.logger.info("The model is trained from scratch")

    # distributed parallel training 
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[args.local_rank], 
        output_device=args.local_rank, 
        find_unused_parameters=True
        )

    min_loss = 1e10 
    
    for epoch in range(start_epoch, args.max_epoch):
        # train for one epoch
        train_sampler.set_epoch(epoch)
        loss = train(args, epoch, train_loader, model, criterion, optimizer)
        args.logger.info("----------Epoch:{:>3d}, training loss is {:>5.4f} ---------". \
                    format(epoch, loss))
        # evaluate on validation set and save model 
        if args.local_rank == 0: 
            if epoch % args.eval_epoch==0 or epoch>=args.max_epoch-5:
                save_checkpoint({
                    'epoch': epoch ,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}, False, args.dump_path)
            if loss<min_loss: 
                min_loss = loss
                save_checkpoint({
                    'epoch': epoch ,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}, True, args.dump_path)
        # learning scheduler step
        scheduler.step()
    
    args.logger.info("------------------- Train Finished -------------------")
