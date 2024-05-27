import time
from utils import *

from transformers import AdamW, get_linear_schedule_with_warmup

def train(args, epoch, train_loader, model, criterion, optimizer, data_collator, scheduler):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        inputs = {k: v.cuda() for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].cuda()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)