import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *


class CrossEntropy(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        loss = F.cross_entropy(output, target)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, cfg=None):
        super(FocalLoss, self).__init__()
        # self.gamma = cfg.LOSS.FOCAL.GAMMA
        if cfg is None:
            self.gamma = 2.0
        else:
            self.gamma = cfg.focal_loss_gamma
        assert self.gamma >= 0

    def focal_loss(self, input_values):
        """Computes the focal loss"""
        p = torch.exp(-input_values)
        loss = (1 - p) ** self.gamma * input_values
        return loss.mean()

    def forward(self, input, target):
        return self.focal_loss(F.cross_entropy(input, target, reduction='none'))

class MaskedCrossEntropy(nn.Module):

    def __init__(self, cfg):
        super(MaskedCrossEntropy, self).__init__()
        self.cfg = cfg
    
    def forward(self, logits, target, length):
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.  B x S x (op_size+const_size+var_size)
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step. B x S
        Returns:
            loss: An average loss value masked by the length.
        """
        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1)) 
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask = sequence_mask(length)
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        return loss

class MLMCrossEntropy(nn.Module):

    def __init__(self, cfg):
        super(MLMCrossEntropy, self).__init__()
        self.cfg = cfg
    
    def forward(self, logits, target):
        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1)) 
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1)
        # losses
        loss = F.cross_entropy(logits_flat, target_flat)

        return loss