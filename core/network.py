import torch
import torch.nn as nn
from model.backbone import get_visual_backbone
from model.encoder import get_encoder, TransformerEncoder
from model.decoder import get_decoder
from utils.utils import *
import numpy as np


class MLMTransformerPretrain(nn.Module):

    def __init__(self, cfg, src_lang):
        super(MLMTransformerPretrain, self).__init__()
        self.cfg = cfg
        self.transformer_en = TransformerEncoder(cfg.encoder_embedding_size)
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
    
    def forward(self, text_dict):
        '''
            text_dict = {'token', 'sect_tag', 'class_tag', 'len'}
        '''
        # text feature
        token_emb = self.text_embedding_src(text_dict['token'])
        class_tag_emb = self.class_tag_embedding(text_dict['class_tag'])
        sect_tag_emb = self.sect_tag_embedding(text_dict['sect_tag'])
        text_emb_src = token_emb.sum(dim=1) + sect_tag_emb + class_tag_emb
        transformer_outputs = self.transformer_en(text_dict['len'], text_emb_src)
        return transformer_outputs

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict_model = pretrain_dict['state_dict'] \
                                if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict_model.items():
            if k in model_dict:
                if k.startswith("module"):
                    new_dict[k[7:]] = v
                else:
                    new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def get_text_embedding_src(self, vocab_size, embedding_dim, padding_idx, pretrain_emb_path):

        embedding_src = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrain_emb_path!='':
            emb_content = []
            with open(pretrain_emb_path, 'r') as f:
                for line in f:
                    emb_content.append(line.split()[1:])
                vector = np.asarray(emb_content, "float32") 
            embedding_src.weight.data[-len(emb_content):]. \
                                    copy_(torch.from_numpy(vector))
        return embedding_src

class Network(nn.Module):
    
    def __init__(self, cfg, src_lang, tgt_lang):
        super(Network, self).__init__()
        self.cfg = cfg
        # define the encoder and decoder
        self.visual_extractor = get_visual_backbone(cfg)  
        self.encoder = get_encoder(cfg)
        self.decoder = get_decoder(cfg, tgt_lang)
        self.visual_emb_unify = nn.ModuleList([
            nn.Linear(self.visual_extractor.final_feat_dim, cfg.encoder_embedding_size), 
            nn.ReLU(),
            nn.Linear(cfg.encoder_embedding_size, cfg.encoder_embedding_size)]
        )
        self.visual_emb_unify = nn.Sequential(*self.visual_emb_unify)

        if cfg.use_MLM_pretrain:
            self.mlm_pretrain = MLMTransformerPretrain(cfg, src_lang)
            if cfg.MLM_pretrain_path!='':
                self.mlm_pretrain.load_model(cfg.MLM_pretrain_path)
        else:
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

        self.src_lang = src_lang

    def forward(self, diagram_src, text_dict, var_dict, exp_dict, is_train=False):
        '''
            diagram_src: B x C x W x H
            text_dict = {'token', 'sect_tag', 'class_tag', 'len'} /
                        {'token', 'sect_tag', 'class_tag', 'subseq_len', 'item_len', 'item_quant'}
            var_dict = {'pos', 'len', 'var_value', 'arg_value'}
            exp_dict = {'exp', 'len', 'answer'}
        '''

        if self.cfg.use_MLM_pretrain:
            text_emb_src = self.mlm_pretrain(text_dict)
        else:
            # text feature
            token_emb = self.text_embedding_src(text_dict['token'])
            class_tag_emb = self.class_tag_embedding(text_dict['class_tag'])
            sect_tag_emb = self.sect_tag_embedding(text_dict['sect_tag'])
            # all feature
            text_emb_src = token_emb.sum(dim=1) + sect_tag_emb + class_tag_emb
        
        # diagram feature
        diagram_emb_src = self.visual_extractor(diagram_src)
        diagram_emb_src = self.visual_emb_unify(diagram_emb_src).unsqueeze(dim=1)
        # feature all
        all_emb_src = torch.cat([diagram_emb_src, text_emb_src], dim=1)
        text_dict['len'] += 1
        var_dict['pos'] += 1
        # encoder
        encoder_outputs, encode_hidden = self.encoder(all_emb_src, text_dict['len'])
        problem_output = encode_hidden[-1:,:,:].repeat(self.cfg.decoder_layers, 1, 1)
        # decoder 
        outputs = self.decoder(encoder_outputs, problem_output, \
                                text_dict['len'], \
                                var_dict['pos'], var_dict['len'], \
                                exp_dict['exp'], \
                                is_train)
        return outputs

    def freeze_module(self, module):
        self.cfg.logger.info("Freezing module of "+" .......")
        for p in module.parameters():
            p.requires_grad = False

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict_model = pretrain_dict['state_dict'] \
                            if 'state_dict' in pretrain_dict else pretrain_dict
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
            embedding_src.weight.data[-len(emb_content):]. \
                                    copy_(torch.from_numpy(vector))
        return embedding_src


def get_model(args, src_lang, tgt_lang):
    model = Network(args, src_lang, tgt_lang)
    args.logger.info(str(model))
    return model


    




