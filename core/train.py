import time
from utils import *

def train(args, epoch, train_loader, model, criterion, optimizer):

    batch_time = AverageMeter('Time', ':5.3f')
    data_time = AverageMeter('Data', ':5.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                             args, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, (diagrams, text_dict, var_dict, exp_dict) in enumerate(train_loader):
        '''
            text_dict = {'token', 'sect_tag', 'class_tag', 'len'}
            var_dict = {'pos', 'len', 'var_value', 'arg_value'}
            exp_dict = {'exp', 'len', 'answer'}
        '''
        # measure data loading time
        data_time.update(time.time() - end)
        # set cuda for input data
        diagrams = diagrams.cuda()
        set_cuda(text_dict), set_cuda(var_dict), set_cuda(exp_dict)
        # compute output
        output = model(diagrams, text_dict, var_dict, exp_dict, is_train=True)
        loss = criterion(output, exp_dict['exp'][:,1:].clone(), exp_dict['len']-1) # Remove special symbol [SOS]
        # update the loss
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args.nprocs)
        losses.update(reduced_loss.item(), len(diagrams))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i, lr = optimizer.state_dict()['param_groups'][0]['lr'])

    return losses.avg
