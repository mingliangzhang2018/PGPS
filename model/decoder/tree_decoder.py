import torch
import torch.nn as nn
from utils import *
from model.module import *
from torch.nn import functional as F
import copy

class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag

class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stacks, embeddings_stacks, left_child_trees, out):
        self.score = score
        self.embeddings_stacks = embeddings_stacks
        self.node_stacks = node_stacks
        self.left_child_trees = left_child_trees
        self.out = out

class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding
    def __init__(self, cfg, op_const_size):
        super(Prediction, self).__init__()
        # Define layers
        self.em_dropout = nn.Dropout(cfg.dropout_rate)
        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(cfg.decoder_hidden_size, cfg.decoder_hidden_size)
        self.concat_r = nn.Linear(cfg.decoder_hidden_size * 2, cfg.decoder_hidden_size)
        self.concat_lg = nn.Linear(cfg.decoder_hidden_size, cfg.decoder_hidden_size)
        self.concat_rg = nn.Linear(cfg.decoder_hidden_size * 2, cfg.decoder_hidden_size)
        # attention module
        self.attn = Attn(cfg.encoder_hidden_size, cfg.decoder_hidden_size)
        self.score = Score(cfg.encoder_hidden_size+cfg.decoder_hidden_size, cfg.decoder_embedding_size)
        # predefined constant
        self.op_const_id = torch.arange(op_const_size).unsqueeze(0).cuda()
        self.padding_hidden = torch.zeros(1, cfg.decoder_hidden_size).cuda()

    def forward(self, node_stacks, left_child_trees, encoder_outputs, var_pades, source_mask, candi_mask, embedding_op_const):
        '''
        Augments:
            node_stacks: [[TreeNode(_)]]*B, store the variable h
            left_child_trees: [t]*B, store the representation of left tree
            encoder_outputs: [B, S1, H]
            var_pades: [B, S2, H], all_vars_encoder_outputs 
            padding_hidden: [1, H]
            source_mask: [B, S1], mask for source seq 
            candi_mask: [B, op_size+const_size+var_size], mask for target seq
        Returns:
            num_score: [B x (op_size+const_size+var_size)]
            current_embeddings: q [B x 1 x H], the target vector of the current node 
            current_context: c [B x 1 x H], the context vector of the current node, is calculated using the target vector and encoder_outputs
            current_all_embeddings: [B x (op_size+const_size+var_size) x H] e (M_op, M_con, h_loc^p) 
        '''
        current_embeddings = []

        for node_list in node_stacks:
            if len(node_list) == 0:
                current_embeddings.append(self.padding_hidden)
            else:
                current_node = node_list[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = [] # B x (1 x H)

        for l, c in zip(left_child_trees, current_embeddings):
            if l is None:
                cd = self.em_dropout(c)
                g = torch.tanh(self.concat_l(cd))
                t = torch.sigmoid(self.concat_lg(cd))
                current_node_temp.append(g*t)
            else:
                ld = self.em_dropout(l)
                cd = self.em_dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, cd), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, cd), 1)))
                current_node_temp.append(g*t) 

        current_node = torch.stack(current_node_temp, dim=0) # B x 1 x H (q)
        current_embeddings = self.em_dropout(current_node) 
        current_attn = self.attn(current_embeddings, encoder_outputs, source_mask) # B x S
        current_context = current_attn.unsqueeze(1).bmm(encoder_outputs)  # B x 1 x H (c)
        leaf_input = torch.cat((current_node, current_context), 2) # B x 1 x 2H
        
        embedding_weight_op_const = embedding_op_const(self.op_const_id.repeat(var_pades.size(0), 1)) # B x var_size x H
        embedding_weight_all = torch.cat((embedding_weight_op_const, var_pades), dim=1)  # B x (op_size+const_size+var_size) x H

        leaf_input = self.em_dropout(leaf_input)
        embedding_weight_all_ = self.em_dropout(embedding_weight_all)   
        num_score = self.score(leaf_input, embedding_weight_all_, candi_mask) # B x (op_size+const_size+var_size)

        return num_score, current_node, current_context, embedding_weight_all

class GenerateNode(nn.Module):
    def __init__(self, cfg, op_size):
        super(GenerateNode, self).__init__()

        self.embedding_size = cfg.decoder_embedding_size
        self.hidden_size = cfg.decoder_hidden_size
        self.op_size = op_size

        self.em_dropout = nn.Dropout(cfg.dropout_rate)
        self.generate_l = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)
        self.generate_r = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)
        self.generate_lg = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)
        self.generate_rg = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)

    def forward(self, current_embedding, node_label, current_context, embedding_op_const):
        """
            Generate the hidden node hl and hr of tree, according to the front part of eq(10)(11)
        Arguments:
            current_embedding: [B x 1 x H (q)], the target vector of the current node
            node_label: [B (id)]
            current_context: [B x 1 x H (c)], context vector of current node
            embedding_op_const: Embedding of op_const
        Returns:
            left_child: [B x H (h)]
            right_child: [B x H (h)]
            token_embedding: [B x H (e(y|P) of op)]
        """
        node_label_op = torch.clamp(node_label, max=self.op_size-1)
        current_embedding_ = self.em_dropout(current_embedding.squeeze(1))
        current_context_ = self.em_dropout(current_context.squeeze(1))
        token_embedding = embedding_op_const(node_label_op)
        token_embedding_ = self.em_dropout(token_embedding)

        l_child = torch.tanh(self.generate_l(torch.cat((current_embedding_, current_context_, token_embedding_), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((current_embedding_, current_context_, token_embedding_), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((current_embedding_, current_context_, token_embedding_), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((current_embedding_, current_context_, token_embedding_), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g

        return l_child, r_child, token_embedding

class Merge(nn.Module):
    """
        Get subtree embedding via Recursive Neural Network
    """
    def __init__(self, cfg):
        super(Merge, self).__init__()

        self.embedding_size = cfg.decoder_embedding_size
        self.hidden_size = cfg.decoder_hidden_size
        self.em_dropout = nn.Dropout(cfg.dropout_rate)
        self.merge = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)
        self.merge_g = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        '''
        Arguments:
            node_embedding: 1 x H
            sub_tree_1: 1 x H 
            sub_tree_2: 1 x H
        Return:
            sub_tree: 1 x H
        '''
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        
        return sub_tree

class TreeDecoder(nn.Module):
    def __init__(self, cfg, tgt_lang):
        super(TreeDecoder, self).__init__()
        # embedding for op, const, num
        self.var_start = tgt_lang.var_start
        self.op_num = tgt_lang.op_num
        self.const_num = tgt_lang.const_num

        self.embedding_op_const = nn.Embedding(self.op_num+self.const_num, cfg.decoder_embedding_size)
        self.embedding_var = None # obtain from encoder
        self.cfg = cfg
        # modules of TreeDecoder 
        self.predict = Prediction(cfg, self.op_num+self.const_num)
        self.generate = GenerateNode(cfg, self.op_num)
        self.merge = Merge(cfg)
    
    def get_var_encoder_outputs(self, encoder_outputs, var_positions):
        """
        Arguments:
            encoder_outputs:  B x S1 x H
            var_positions: B x S2
        Returns:
            var_embeddings: B x S2 x H
        """
        hidden_size = encoder_outputs.size(-1)
        expand_var_positions = var_positions.unsqueeze(-1).repeat(1, 1, hidden_size)
        var_embeddings = encoder_outputs.gather(dim=1, index = expand_var_positions)
        return var_embeddings

    def forward(self, encoder_outputs, problem_output, len_source, var_positions, len_var, \
                            is_train=False, text_target=None, len_target=None):
        """
        Arguments:
            encoder_outputs: B x S1 x H
            problem_output: B x H
            len_source: B
            text_target: B x S2
            len_target: B
            var_positions: B x S3
            len_var: B
        Return:
            training: output B x S x (op_size+const_size+var_size), logits of one batch
            testing: [expr] x B
        """
        self.embedding_var = self.get_var_encoder_outputs(encoder_outputs, var_positions) # B x S2 x H
        self.source_mask = sequence_mask(len_source)
        self.candi_mask = sequence_mask(len_var+self.var_start)
        if is_train:
            return self._forward_train(encoder_outputs, problem_output, text_target)  
        else:
            return self._forward_test(encoder_outputs, problem_output)
            
    def _forward_train(self, encoder_outputs, problem_output, text_target):
        """
        Arguments:
            embeddings_stacks: [[TreeEmbedding(t, terminal)]]*B, a stack of subtrees t in the first order traversal
            left_child_trees: [t]*B, the representation of left tree of current node
            node_stacks: [[TreeNode(h, left_flag)]]*B, a stack of hidden state h in the first order traversal
        Returns:
            all_node_outputs: B x S x (op_size+const_size+var_size), logits of one batch
        """
        node_stacks = [[TreeNode(init_hidden)] for init_hidden in problem_output.split(1, dim=0)]
        embeddings_stacks = [[] for _ in range(encoder_outputs.size(0))]
        left_child_trees = [None]*encoder_outputs.size(0)
        all_node_outputs = []

        for t in range(text_target.size(1)):
            num_score, current_embeddings, current_context, current_all_embeddings = self.predict(
                    node_stacks, 
                    left_child_trees, 
                    encoder_outputs, 
                    self.embedding_var, 
                    self.source_mask, 
                    self.candi_mask,
                    self.embedding_op_const)

            all_node_outputs.append(num_score) # [B x (op_size+const_size+var_size)] * S

            left_child, right_child, token_embedding = self.generate(
                    current_embeddings, 
                    text_target[:,t], 
                    current_context, 
                    self.embedding_op_const)

            left_child_trees = []

            for idx, (l, r, node_stack, target_id, embeddings_stack) in enumerate(zip(left_child.split(1), right_child.split(1),
                                                                        node_stacks, text_target[:,t].tolist(), embeddings_stacks)):
                # Determines whether the tree traversal is complete                                                    
                if len(node_stack) != 0:
                    node_stack.pop()
                else:
                    left_child_trees.append(None)
                    continue
                if target_id < self.op_num:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    # embeddings_stack, put e(y|P) of op in temporarily
                    embeddings_stack.append(TreeEmbedding(token_embedding[idx].unsqueeze(0), False))
                else:
                    current_num = current_all_embeddings[idx, target_id].unsqueeze(0) # 1 x H
                    # Reach the right leaf node and merge the tree representation from bottom up
                    while len(embeddings_stack) > 0 and embeddings_stack[-1].terminal: 
                        sub_stree = embeddings_stack.pop()
                        op = embeddings_stack.pop()
                        # embedding vector of two sub-targets is merged as the subtree embedding of nodes, corresponding to eq(12)
                        # with e(y|P), sub_tree_1 and sub_tree_2
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                    embeddings_stack.append(TreeEmbedding(current_num, True))
                # Reach the left leaf node and save the representation of the left subtree for generation of q
                if len(embeddings_stack) > 0 and embeddings_stack[-1].terminal: 
                    left_child_trees.append(embeddings_stack[-1].embedding)
                else:
                    left_child_trees.append(None)

        all_node_outputs = torch.stack(all_node_outputs, dim=1)  

        return all_node_outputs

    def _forward_test(self, encoder_outputs, problem_output):

        exp_outputs = [] 

        for sample_id in range(encoder_outputs.size(0)):
            # set batch size as 1
            node_stacks = [[TreeNode(problem_output[sample_id:sample_id+1])]]
            embeddings_stacks = [[]]
            left_child_trees = [None]
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_child_trees, [])]

            for _ in range(self.cfg.max_output_len):
                # re-maintain of one beams
                current_beams = []

                while len(beams) > 0:
                    beam_item = beams.pop()
                    # The candidates are stored in beams in all process
                    if len(beam_item.node_stacks[0]) == 0:
                        current_beams.append(beam_item) 
                        continue
                    num_score, current_embeddings, current_context, current_all_embeddings = self.predict(
                            beam_item.node_stacks, 
                            beam_item.left_child_trees, 
                            encoder_outputs[sample_id:sample_id+1], 
                            self.embedding_var[sample_id:sample_id+1], 
                            self.source_mask[sample_id:sample_id+1], 
                            self.candi_mask[sample_id:sample_id+1],
                            self.embedding_op_const)

                    out_score = F.log_softmax(num_score, dim=1)
                    topv, topi = out_score.topk(self.cfg.beam_size)

                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):

                        current_node_stack = copy_list(beam_item.node_stacks)
                        current_left_child_trees = []
                        current_embeddings_stacks = copy_list(beam_item.embeddings_stacks)
                        current_out = copy.deepcopy(beam_item.out)

                        out_token = int(ti)
                        current_out.append(out_token)
                        current_node_stack[0].pop()

                        if out_token < self.op_num:
                            generate_input = torch.LongTensor([out_token]).cuda()
                            left_child, right_child, token_embedding = self.generate(
                                current_embeddings, 
                                generate_input, 
                                current_context, 
                                self.embedding_op_const)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(token_embedding, False))
                        else:
                            current_num = current_all_embeddings[:, out_token]
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_child_trees.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_child_trees.append(None)
                        
                        current_beams.append(TreeBeam(beam_item.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                    current_left_child_trees, current_out))
                
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.cfg.beam_size]

                # early termination
                flag = True
                for beam_item in beams:
                    if len(beam_item.node_stacks[0]) != 0:
                        flag = False
                        break 
                if flag: break

            exp_outputs.append(beams[0].out)

        return exp_outputs
