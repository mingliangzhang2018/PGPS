import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
from core.train import *
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
    else:
        args.logger.info("The model is trained from scratch")

    # distributed parallel training 
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[args.local_rank], 
        output_device=args.local_rank, 
        find_unused_parameters=True
        )
    
    for epoch in range(start_epoch, args.max_epoch):
        # train for one epoch
        train_sampler.set_epoch(epoch)
        loss = train(args, epoch, train_loader, model, criterion, optimizer)
        args.logger.info("----------Epoch:{:>3d}, training loss is {:>5.4f} ---------". \
                    format(epoch, loss))
        # evaluate on validation set and save model  
        if epoch>0 and (epoch % args.eval_epoch ==0 or epoch>=args.max_epoch-5):
            is_best = False
            if args.local_rank == 0: 
                save_checkpoint({
                        'epoch': epoch ,
                        'state_dict': model.module.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()}, is_best, args.dump_path)
        # learning scheduler step
        scheduler.step()
    
    args.logger.info("------------------- Train Finished -------------------")
