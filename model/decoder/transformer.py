import torch
import torch.nn as nn
from utils.utils import sequence_mask
from model.module import *
from torch.nn import functional as F
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout_rate=0.2):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
            x: [B, max_len, d_model]
            pe: [1, max_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TransformerDecoder(nn.Module):

    def __init__(self, cfg, tgt_lang, \
                d_model=256, nhead=8, num_decoder_layers=4, dim_feedforward=1024, dropout=0.2):
        super(TransformerDecoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.position_dec = PositionalEncoding(d_model=d_model)

        self.score = Score_Multi(cfg.decoder_hidden_size, cfg.decoder_embedding_size)
        self.var_start = tgt_lang.var_start
        self.embedding_tgt = nn.Embedding(self.var_start, cfg.decoder_embedding_size, padding_idx=0)
        self.no_var_id = torch.arange(self.var_start).unsqueeze(0).cuda()

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.cfg = cfg
        self.sos_id = tgt_lang.word2index["[SOS]"]
        self.eos_id = tgt_lang.word2index["[EOS]"]
        
    def _reset_parameters(self):
        """
            Initiate parameters in the transformer model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_square_subsequent_mask(self, sz):
        """
            Generate a square mask for the sequence. The masked positions are filled with True.
            Unmasked positions are filled with False.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 0).transpose(0, 1)
        return mask.cuda()
    
    def get_var_encoder_outputs(self, encoder_outputs, var_pos):
        """
        Arguments:
            encoder_outputs:  B x S1 x H
            var_pos: B x S3
        Returns:
            var_embeddings: B x S3 x H
        """
        hidden_size = encoder_outputs.size(-1)
        expand_var_pos = var_pos.unsqueeze(-1).repeat(1, 1, hidden_size)
        var_embeddings = encoder_outputs.gather(dim=1, index = expand_var_pos)
        return var_embeddings
    
    def forward(self, memory, len_src, tgt, len_tgt, var_pos, len_var, is_train=False):
        '''
            memory: B x S1 x H
            len_src: B
            tgt: B x S2
            len_tgt: B
            var_pos: B x S3(var_size)
            len_var: B
        '''
        self.embedding_var = self.get_var_encoder_outputs(memory, var_pos) # B x S3 x H
        self.candi_mask = sequence_mask(self.var_start + len_var) # B x (no_var_size + var_size)
        self.memory_key_padding_mask = ~sequence_mask(len_src) # B x S1
        if is_train:
            return self._forward_train(memory, tgt, len_tgt)  
        else:
            return self._forward_test(memory)


    def _forward_train(self, memory, tgt, len_tgt):
        # mask
        tgt_mask = self.get_square_subsequent_mask(tgt.size(-1))
        tgt_key_padding_mask = ~sequence_mask(len_tgt)
        # emb_tgt
        tgt_novar_id = torch.clamp(tgt, max=self.var_start-1) # B x S2
        novar_embedding = self.embedding_tgt(tgt_novar_id) # B x S2 x H
        tgt_var_id = torch.clamp(tgt-self.var_start, min=0) # B x S2
        var_embeddings = self.embedding_var.gather(dim=1, index = \
                            tgt_var_id.unsqueeze(2).repeat(1, 1, self.cfg.decoder_embedding_size)) # B x S2 x H
        choose_mask = (tgt<self.var_start).unsqueeze(2). \
                                repeat(1, 1, self.cfg.decoder_embedding_size)
        emb_tgt = torch.where(choose_mask, novar_embedding, var_embeddings) # B x S2 x H
        # position decoding
        emb_tgt = self.position_dec(emb_tgt)
        output = self.decoder( # B x S2 x H
            emb_tgt.permute(1,0,2), 
            memory.permute(1,0,2), 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=self.memory_key_padding_mask,
        ).permute(1,0,2)
        # candi weight embedding
        embedding_weight_no_var = self.embedding_tgt(self.no_var_id.repeat(len(len_tgt), 1)) # B x no_var_size x H
        embedding_weight_all = torch.cat((embedding_weight_no_var, self.embedding_var), dim=1)  # B x (no_var_size+var_size) x H
        candi_score = self.score( #  B x S2 x (no_var_size + var_size)
            output, 
            embedding_weight_all, \
            self.candi_mask
            ) 

        return candi_score[:,:-1,:].clone()

    def _forward_test(self, memory):
        
        exp_outputs = []

        for sample_id in range(memory.size(0)):
            # predefine 
            rem_size = self.cfg.beam_size
            memory_item = memory[sample_id:sample_id+1].repeat(rem_size, 1, 1) # beam_size x S1 x H
            memory_key_padding_mask = self.memory_key_padding_mask[sample_id:sample_id+1].repeat(rem_size, 1) # beam_size x S1
            embedding_var = self.embedding_var[sample_id:sample_id+1].repeat(rem_size, 1, 1) # beam_size x S3 x H
            embedding_weight_no_var = self.embedding_tgt(self.no_var_id.repeat(rem_size, 1)) # beam_size x no_var_size x H
            embedding_weight_all = torch.cat((embedding_weight_no_var, embedding_var), dim=1)  # beam_size x (no_var_size + var_size) x H
            candi_mask = self.candi_mask[sample_id:sample_id+1].repeat(rem_size, 1) # beam_size x S1

            candi_exp_output = []
            candi_score_output = []
            
            tgt = torch.LongTensor([[self.sos_id]]*rem_size).cuda() # rem_size x 1
            len_tgt = torch.LongTensor([1]*rem_size).cuda() # rem_size
            current_score = torch.FloatTensor([[0.0]]*rem_size).cuda() # rem_size x 1
            current_exp_list = [[self.sos_id]]*rem_size

            for i in range(self.cfg.max_output_len):
                # mask
                tgt_mask = self.get_square_subsequent_mask(tgt.size(-1))
                tgt_key_padding_mask = ~sequence_mask(len_tgt)
                # input embedding
                tgt_novar_id = torch.clamp(tgt, max=self.var_start-1) # rem_size x S
                novar_embedding = self.embedding_tgt(tgt_novar_id) # rem_size x S x H
                tgt_var_id = torch.clamp(tgt-self.var_start, min=0) # rem_size x S
                var_embeddings = embedding_var[:rem_size].gather(dim=1, index=tgt_var_id.unsqueeze(2). \
                                                    repeat(1, 1, self.cfg.decoder_embedding_size)) # rem_size x S x H
                choose_mask = (tgt<self.var_start).unsqueeze(2).repeat(1, 1, self.cfg.decoder_embedding_size) # rem_size x S x H
                emb_tgt = torch.where(choose_mask, novar_embedding, var_embeddings) # rem_size x S x H
                # position decoding
                emb_tgt = self.position_dec(emb_tgt)
                output = self.decoder( # rem_size x S x H
                    emb_tgt.permute(1,0,2), 
                    memory_item[:rem_size].permute(1,0,2), 
                    tgt_mask=tgt_mask, 
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask[:rem_size],
                ).permute(1,0,2)
                candi_score = self.score( # rem_size x S x (no_var_size + var_size)
                    output, 
                    embedding_weight_all[:rem_size], \
                    candi_mask[:rem_size]
                    ) 

                if i==0:
                    new_score = F.log_softmax(candi_score[:, -1, :], dim=1)[:1]
                else:
                    new_score = F.log_softmax(candi_score[:, -1, :], dim=1) + current_score # rem_size x (no_var_size + var_size)
                
                topv, topi = new_score.view(-1).topk(rem_size)
                exp_list = []
                score_list = topv.tolist()

                for tv, ti in zip(topv, topi):
                    idex = ti.item()
                    x = idex // candi_score.size(-1) 
                    y = idex % candi_score.size(-1)
                    if y!=self.eos_id:
                        exp_list.append(current_exp_list[x]+[y])
                    else:
                        candi_exp_output.append(current_exp_list[x][1:])
                        candi_score_output.append(float(tv))

                if len(exp_list)==0:
                    break

                tgt = torch.LongTensor(exp_list).cuda() # rem_size x S
                len_tgt = torch.LongTensor([len(item) for item in exp_list]).cuda() # rem_size
                current_exp_list = exp_list
                rem_size = len(exp_list)
                current_score = torch.FloatTensor(score_list[:rem_size]).unsqueeze(1).cuda() # rem_size x 1
  
            if len(candi_exp_output)>0:
                _, candi_exp_output = zip(*sorted(zip(candi_score_output, candi_exp_output), reverse=True))
                exp_outputs.append(list(candi_exp_output))
            else:
                exp_outputs.append([])

        return exp_outputs
