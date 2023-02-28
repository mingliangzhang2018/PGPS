import torch
import json
from datasets.utils import *

class SrcLang:

    def __init__(self, vocab_path):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0
        self.get_vocab(vocab_path)
        self.class_tag = ['[PAD]', '[GEN]', '[POINT]', '[NUM]', '[ARG]', '[ANGID]']
        self.sect_tag = ['[PAD]', '[PROB]', '[COND]', '[STRU]']
        
    def get_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            for id, line in enumerate(f):
                vocab_token = line[:-1]
                self.word2index[vocab_token] = id
                self.word2count[vocab_token] = 0
                self.index2word.append(vocab_token)
        self.n_words = len(self.index2word)
    
    def indexes_from_sentence(self, sentence, id_type='text'):
        res = []
        if id_type == 'text':
            for word in sentence:
                if word in self.word2index:
                    res.append(self.word2index[word])
                    self.word2count[word] += 1
                else:
                    res.append(self.word2index["[UNK]"])
                    self.word2count["[UNK]"] += 1
                    print("Can not find", word, 'in the src vocab')
        elif id_type=='class_tag':
            for word in sentence: res.append(self.class_tag.index(word))
        elif id_type=='sect_tag':
            for word in sentence: res.append(self.sect_tag.index(word))
        return res
    
    def sentence_from_indexes(self, indexes):
        res = []
        for index in indexes:
            if index<len(self.index2word):
                res.append(self.index2word[index])
            else:
                res.append("")
        return res

class TgtLang:

    def __init__(self, vocab_path):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0 
        self.var_start = 0
        self.get_vocab(vocab_path)
    
    def get_vocab(self, vocab_path):
        spe_num = midvar_num = const_num = 0
        op_num = var_num = 0
        
        with open(vocab_path, 'r') as f:
            for id, line in enumerate(f):
                vocab_token = line[:-1]
                self.word2index[vocab_token] = id
                self.word2count[vocab_token] = 0
                self.index2word.append(vocab_token)
                if vocab_token[0]=='[' and vocab_token[-1]==']': 
                    spe_num += 1
                elif vocab_token[0]=='V' and vocab_token[1].isdigit(): 
                    midvar_num += 1
                elif vocab_token[0]=='C' and vocab_token[1].isdigit(): 
                    const_num += 1
                elif vocab_token[0]=='N' and vocab_token[1].isdigit():
                    var_num += 1
                else:
                    op_num += 1

        self.n_words = len(self.index2word)
        self.var_start = spe_num + midvar_num + const_num + op_num

    def indexes_from_sentence(self, sentence, var_values, arg_values):
        res = []
        for word in sentence:
            if word in self.word2index:
                res.append(self.word2index[word])
                self.word2count[word] += 1
            elif len(word)==1 and word.islower(): # arg
                res.append(self.var_start+len(var_values)+arg_values.index(word))
            else:
                print("Can not find", word, 'in the tgt vocab')
        res = [self.word2index["[SOS]"]]+res+[self.word2index["[EOS]"]]
        return res
    
    def sentence_from_indexes(self, indexes, change_dict={}):
        res = []
        for index in indexes:
            if index<len(self.index2word):
                item = self.index2word[index]
            else:
                item = ''
            if item in change_dict: item = change_dict[item] # var2arg
            res.append(item)
        return res
    
class SN:
    def __init__(self):
        self.token = [] # str list
        self.sect_tag = [] # [PROB]/[COND]/[STRU]
        self.class_tag = [] # [GEN]/[NUM]/[ARG]/[POINT]/[ANGID]

def get_raw_pairs(dataset_path):

    raw_pairs = []

    with open(dataset_path, 'r')as fp:
        content_all = json.load(fp)

    for key, content in content_all.items():
        text = content['text']
        stru_seqs = content['parsing_stru_seqs']
        sem_seqs = content['parsing_sem_seqs']
        text_data, stru_data, sem_data = SN(), SN(), SN()
        # tokenization
        text_data.token = get_token(text)
        stru_data.token = [get_token(item)+[','] for item in stru_seqs]
        sem_data.token = [get_token(item)+[','] for item in sem_seqs]
        # split prob and cond
        text_data.sect_tag = []
        stru_data.sect_tag = [['[STRU]']*len(item) for item in stru_data.token]
        sem_data.sect_tag = [['[COND]']*len(item) for item in sem_data.token]
        split_text(text_data)
        # get class tag
        text_data.class_tag = ['[GEN]']*len(text_data.token)
        stru_data.class_tag = [['[GEN]']*len(item) for item in stru_data.token]
        sem_data.class_tag = [['[GEN]']*len(item) for item in sem_data.token]
        get_point_angleID_tag(text_data, stru_data, sem_data)
        get_num_arg_tag(text_data, sem_data)
        # Tag the repeat [NUM] in sem_data which has exist in text_data
        expression = content['expression'].split(' ')
        remove_sem_dup(text_data, sem_data, expression)

        content['text'] = text_data
        content['parsing_stru_seqs'] = stru_data
        content['parsing_sem_seqs'] = sem_data
        content['expression'] = expression
        content['id'] = key
        
        raw_pairs.append(content)
        
    return raw_pairs

class collater():

    def __init__(self, args, src_lang):
        self.args = args
        self.src_lang = src_lang

    def __call__(self, batch_data, padding_id=0):

        text_tokens, text_sect_tags, text_class_tags = list(zip(*batch_data))

        len_text = [len(seq_tag) for seq_tag in text_class_tags]
        max_len_text = max(len_text)

        for k in range(len(text_tokens)):
            for j in range(len(text_tokens[k])):
                text_tokens[k][j] += [padding_id]*(max_len_text-len(text_tokens[k][j]))
        text_sect_tags = [seq_tag+[padding_id]*(max_len_text-len(seq_tag)) for seq_tag in text_sect_tags]
        text_class_tags = [seq_tag+[padding_id]*(max_len_text-len(seq_tag)) for seq_tag in text_class_tags]

        text_tokens = torch.LongTensor(text_tokens)
        text_sect_tags = torch.LongTensor(text_sect_tags)
        text_class_tags = torch.LongTensor(text_class_tags)
        len_text = torch.LongTensor(len_text)
        
        text_tokens, labels = get_mask_tokens(text_tokens, text_class_tags, \
                                                self.src_lang, self.args.mlm_probability)

        text_dict = {
            'token': text_tokens,
            'sect_tag': text_sect_tags,
            'class_tag': text_class_tags,
            'len': len_text,
            'labels': labels}

        return text_dict

def get_mask_tokens(text_tokens, text_class_tags, src_lang, mlm_probability):

    prob_replace_mask=0.8
    prob_replace_rand=0.1
    prob_keep_ori=0.1

    labels = text_tokens[:,0].clone()
    inputs = text_tokens[:,0].clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    special_tokens_mask = text_class_tags==src_lang.class_tag.index('[PAD]')
    # special_tokens_mask = text_class_tags!=src_lang.class_tag.index('[POINT]')
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, prob_replace_mask)).bool() & masked_indices
    inputs[indices_replaced] = src_lang.word2index['[MASK]']

    # 10% of the time, we replace masked input tokens with random word
    current_prob = prob_replace_rand / (1 - prob_replace_mask)
    indices_random = torch.bernoulli(torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(src_lang.index2word), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    text_tokens = torch.stack((inputs, text_tokens[:,1]), dim=1)
    
    return text_tokens, labels
