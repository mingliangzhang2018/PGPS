import torch
import torch.nn as nn
from model.encoder import TransformerEncoder
from utils.utils import *
import numpy as np

class Network(nn.Module):
    
    def __init__(self, cfg, src_lang, tgt_lang):
        super(Network, self).__init__()
        self.cfg = cfg
        self.transformer_en = TransformerEncoder(cfg.encoder_hidden_size)
        self.text_embedding_src = self.get_text_embedding_src(
            vocab_size = src_lang.n_words,
            embedding_dim = cfg.encoder_embedding_size,
            padding_idx = 0,
            pretrain_emb_path = cfg.pretrain_emb_path
        )
        self.class_tag_embedding = nn.Embedding(
            len(src_lang.class_tag), 
            cfg.encoder_embedding_size, 
            padding_idx=0
        )
        self.sect_tag_embedding = nn.Embedding(
            len(src_lang.sect_tag), 
            cfg.encoder_embedding_size, 
            padding_idx=0
        )
        self.score = nn.Linear(cfg.encoder_hidden_size, src_lang.n_words)

    def forward(self, text_dict, is_train=True):
        '''
            text_dict = {'token', 'sect_tag', 'class_tag', 'len', 'labels'}
        '''
        token_emb = self.text_embedding_src(text_dict['token'])
        class_tag_emb = self.class_tag_embedding(text_dict['class_tag'])
        sect_tag_emb = self.sect_tag_embedding(text_dict['sect_tag'])
        text_emb_src = token_emb.sum(dim=1) + sect_tag_emb + class_tag_emb
        transformer_outputs = self.transformer_en(text_dict['len'], text_emb_src)
        return self.score(transformer_outputs)

    def freeze_module(self, module):
        self.cfg.logger.info("Freezing module of "+" .......")
        for p in module.parameters():
            p.requires_grad = False

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict_model = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict_model.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        return pretrain_dict
    
    def get_text_embedding_src(self, vocab_size, embedding_dim, padding_idx, pretrain_emb_path):
        embedding_src = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrain_emb_path!='':
            emb_content = []
            with open(pretrain_emb_path, 'r') as f:
                for line in f:
                    emb_content.append(line.split()[1:])
                vector = np.asarray(emb_content, "float32") 
            embedding_src.weight.data[-len(emb_content):].copy_(torch.from_numpy(vector))
        return embedding_src
    
def get_model(args, src_lang, tgt_lang):
    model = Network(args, src_lang, tgt_lang)
    args.logger.info(str(model))
    return model





    
