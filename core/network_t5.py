import torch
import torch.nn as nn
from model.backbone import get_visual_backbone
from model.encoder import get_encoder, TransformerEncoder
from model.decoder import get_decoder
from utils.utils import *
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_model(args, src_lang, tgt_lang):
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    args.tokenizer = tokenizer
    args.logger.info(str(model))
    return model