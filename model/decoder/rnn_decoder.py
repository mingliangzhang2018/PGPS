import torch
import torch.nn as nn
from model.module import *
from utils import *
from torch.nn import functional as F

class DecoderRNN(nn.Module):
    def __init__(self, cfg, tgt_lang):
        super(DecoderRNN, self).__init__()
        # token location
        self.var_start = tgt_lang.var_start # spe_num + midvar_num + const_num + op_num
        self.sos_id = tgt_lang.word2index["[SOS]"]
        self.eos_id = tgt_lang.word2index["[EOS]"]
        # Define layers
        self.em_dropout = nn.Dropout(cfg.dropout_rate)
        self.embedding_tgt = nn.Embedding(self.var_start, cfg.decoder_embedding_size, padding_idx=0)
        self.gru = nn.GRU(input_size=cfg.decoder_hidden_size+cfg.decoder_embedding_size, \
                            hidden_size=cfg.decoder_hidden_size, \
                            num_layers=cfg.decoder_layers, \
                            dropout = cfg.dropout_rate, \
                            batch_first = True)
        # Choose attention model
        self.attn = Attn(cfg.encoder_hidden_size, cfg.decoder_hidden_size)
        self.score = Score(cfg.encoder_hidden_size+cfg.decoder_hidden_size, cfg.decoder_embedding_size)
        # predefined constant
        self.no_var_id = torch.arange(self.var_start).unsqueeze(0).cuda()
        self.cfg = cfg
    
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
    
    def forward(self, encoder_outputs, problem_output, len_src, var_pos, len_var, \
                            text_tgt=None, is_train=False):
        """
        Arguments:
            encoder_outputs: B x S1 x H
            problem_output: layer_num x B x H
            len_src: B
            text_tgt: B x S2
            var_pos: B x S3
            len_var: B
        Return:
            training: logits, B x S x (no_var_size+var_size)
            testing: exp_id, B x candi_size(beam_size) x exp_len
        """
        self.embedding_var = self.get_var_encoder_outputs(encoder_outputs, var_pos) # B x S3 x H
        self.src_mask = sequence_mask(len_src)  # B x S1
        self.candi_mask = sequence_mask(self.var_start + len_var) # B x (no_var_size + var_size)
        if is_train:
            return self._forward_train(encoder_outputs, problem_output, text_tgt)  
        else:
            return self._forward_test(encoder_outputs, problem_output)
            
    def _forward_train(self, encoder_outputs, problem_output, text_tgt):

        all_seq_outputs = []
        batch_size = encoder_outputs.size(0)
        # initial hidden input of RNN
        rnn_hidden = problem_output
        # input embedding
        tgt_novar_id = torch.clamp(text_tgt, max=self.var_start-1) # B x S2
        novar_embedding = self.embedding_tgt(tgt_novar_id) # B x S2 x H
        tgt_var_id = torch.clamp(text_tgt-self.var_start, min=0) # B x S2
        var_embeddings = self.embedding_var.gather(dim=1, index = \
                            tgt_var_id.unsqueeze(2).repeat(1, 1, self.cfg.decoder_embedding_size)) # B x S2 x H

        choose_mask = (text_tgt<self.var_start).unsqueeze(2). \
                                repeat(1, 1, self.cfg.decoder_embedding_size)
        embedding_all = torch.where(choose_mask, novar_embedding, var_embeddings) # B x S2 x H
        embedding_all_ = self.em_dropout(embedding_all)
        embedding_weight_no_var = self.embedding_tgt(self.no_var_id. \
                                    repeat(batch_size, 1)) # B x no_var_size x H
        embedding_weight_all = torch.cat((embedding_weight_no_var, self.embedding_var), dim=1) # B x (no_var_size + var_size) x H
        embedding_weight_all_ = self.em_dropout(embedding_weight_all) 

        for t in range(text_tgt.size(1)-1):
            # Calculate attention from current RNN state and all encoder outputs;
            # apply to encoder outputs to get weighted average
            current_hiddens = self.em_dropout(rnn_hidden[-1].unsqueeze(1)) # B x 1 x H
            attn_weights = self.attn(current_hiddens, encoder_outputs, self.src_mask)
            context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # B x 1 x H 
            # Get current hidden state from input word and last hidden state
            rnn_output, rnn_hidden = self.gru(torch.cat((embedding_all_[:, t:t+1, :], context), 2), rnn_hidden) 
            # rnn_output: B x 1 x H 
            # rnn_hidden: num_layers x B x H 
            current_fusion_emb = torch.cat((rnn_output, context), 2)
            current_fusion_emb_ = self.em_dropout(current_fusion_emb)
            candi_score = self.score(current_fusion_emb_, embedding_weight_all_, \
                                            self.candi_mask) #  B x (no_var_size + var_size)
            all_seq_outputs.append(candi_score)
        
        all_seq_outputs = torch.stack(all_seq_outputs, dim=1)  

        return all_seq_outputs
    
    def _forward_test(self, encoder_outputs, problem_output):
        """
            Decode with beam search algorithm
        """
        exp_outputs = []
        batch_size = encoder_outputs.size(0)

        for sample_id in range(batch_size):
            # predefine 
            rem_size = self.cfg.beam_size
            encoder_output = encoder_outputs[sample_id:sample_id+1].repeat(rem_size, 1, 1) # beam_size x S1 x H
            src_mask = self.src_mask[sample_id:sample_id+1].repeat(rem_size, 1) # beam_size x S1
            embedding_var = self.embedding_var[sample_id:sample_id+1].repeat(rem_size, 1, 1) # beam_size x S3 x H
            embedding_weight_no_var = self.embedding_tgt(self.no_var_id.repeat(rem_size, 1)) # beam_size x no_var_size x H
            embedding_weight_all = torch.cat((embedding_weight_no_var, embedding_var), dim=1)  # beam_size x (no_var_size + var_size) x H
            embedding_weight_all_ = self.em_dropout(embedding_weight_all) 
            candi_mask = self.candi_mask[sample_id:sample_id+1].repeat(rem_size, 1) # beam_size x S1
            candi_exp_output = []
            candi_score_output = []
            
            for i in range(self.cfg.max_output_len):
                # initial varible
                if i==0:
                    input_token = torch.LongTensor([[self.sos_id]]*rem_size).cuda() # rem_size x 1
                    rnn_hidden = problem_output[:, sample_id:sample_id+1].repeat(1, rem_size, 1) # layer_num x rem_size x H
                    current_score = torch.FloatTensor([[0.0]]*rem_size).cuda() # rem_size x 1
                    current_exp_list = [[]]*rem_size
                else:
                    input_token = torch.LongTensor(token_list).unsqueeze(1).cuda() 
                    rnn_hidden = rnn_hidden[:, cand_list]
                    rem_size = len(exp_list)
                    current_score = torch.FloatTensor(score_list[:rem_size]).unsqueeze(1).cuda()
                    current_exp_list = exp_list
                # input embedding
                tgt_novar_id = torch.clamp(input_token, max=self.var_start-1) # rem_size x 1
                novar_embedding = self.embedding_tgt(tgt_novar_id) # rem_size x 1 x H
                tgt_var_id = torch.clamp(input_token-self.var_start, min=0) # rem_size x 1
                var_embeddings = embedding_var[:rem_size].gather(dim=1, index=tgt_var_id.unsqueeze(2). \
                                            repeat(1, 1, self.cfg.decoder_embedding_size)) # rem_size x 1 x H
                choose_mask = (input_token<self.var_start).unsqueeze(2). \
                                        repeat(1, 1, self.cfg.decoder_embedding_size) # rem_size x 1 x H
                embedding_all = torch.where(choose_mask, novar_embedding, var_embeddings) # rem_size x 1 x H
                embedding_all_ = self.em_dropout(embedding_all)
                # attention 
                current_hiddens = self.em_dropout(rnn_hidden[-1].unsqueeze(1))  # rem_size x 1 x H
                attn_weights = self.attn(current_hiddens, encoder_output[:rem_size], src_mask[:rem_size]) # rem_size x S1 
                context = attn_weights.unsqueeze(1).bmm(encoder_output[:rem_size])  # rem_size x 1 x H 
                # Get current hidden state from input word and last hidden state
                rnn_output, rnn_hidden = self.gru(torch.cat((embedding_all_, context), 2), rnn_hidden)
                # rnn_output: rem_size x 1 x H 
                # rnn_hidden: num_layers x rem_size x H 
                current_fusion_emb = torch.cat((rnn_output, context), 2)
                current_fusion_emb_ = self.em_dropout(current_fusion_emb)
                candi_score = self.score(current_fusion_emb_, embedding_weight_all_[:rem_size], \
                                                candi_mask[:rem_size]) #  rem_size x (no_var_size + var_size)
                
                if i==0:
                    new_score = F.log_softmax(candi_score, dim=1)[:1]
                else:
                    new_score = F.log_softmax(candi_score, dim=1) + current_score

                cand_tup_list = [(score, id) for id, score in enumerate(new_score.view(-1).tolist())]
                cand_tup_list += [(score, -1) for score in candi_score_output]
                cand_tup_list.sort(key=lambda x:x[0], reverse=True)

                token_list = []
                cand_list = []
                exp_list = []
                score_list = []

                for tv, ti in cand_tup_list[:self.cfg.beam_size]:
                    if ti!=-1:
                      idex = ti
                      x = idex // candi_score.size(-1) 
                      y = idex % candi_score.size(-1)
                      if y!=self.eos_id:
                          token_list.append(y)
                          cand_list.append(x)
                          exp_list.append(current_exp_list[x]+[y])
                          score_list.append(tv)
                      else:
                          candi_exp_output.append(current_exp_list[x])
                          candi_score_output.append(float(tv))

                if len(token_list)==0:
                    break

            if len(candi_exp_output)>0:
                _, candi_exp_output = zip(*sorted(zip(candi_score_output, candi_exp_output), reverse=True))
                exp_outputs.append(list(candi_exp_output[:self.cfg.beam_size]))
            else:
                exp_outputs.append([])

        return exp_outputs
