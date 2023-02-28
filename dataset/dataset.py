import torch
import datasets.text_aug as T_text
from datasets.utils import get_combined_text, get_text_index
from datasets.preprossing import SN

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, args, pairs, src_lang, tgt_lang, is_train=True):
        super().__init__()
        self.args = args
        self.pairs = pairs
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        if is_train: 
            random_prob = args.random_prob
        else:
            random_prob = 0
        self.text_transform = T_text.Compose([
            T_text.Point_RandomReplace(random_prob),
            T_text.AngID_RandomReplace(random_prob),
            T_text.Arg_RandomReplace(random_prob),
            T_text.StruPoint_RandomRotate(random_prob),
            T_text.SemPoint_RandomRotate(random_prob),
            T_text.SemSeq_RandomRotate(random_prob),
            T_text.StruSeq_RandomRotate(random_prob),
        ])
        
    def __getitem__(self, idx):
        '''
            pair{
                'diagram': str
                'text': SN()
                'parsing_stru_seqs': SN()
                'parsing_sem_seqs': SN()
                'expression': list
                'answer': str
                }
        '''
        pair = self.pairs[idx]
        # text, parsing_stru_seqs, parsing_sem_seqs, 
        self.text_transform(pair['text'], 
                            pair['parsing_stru_seqs'], 
                            pair['parsing_sem_seqs'],
                            pair['expression'])
        combine_text = SN()
        get_combined_text(pair['text'], 
                            pair['parsing_stru_seqs'], 
                            pair['parsing_sem_seqs'],
                            combine_text,
                            self.args)
        text_token, text_sect_tag, text_class_tag = \
                            get_text_index(combine_text, self.src_lang)
        
        return  text_token, text_sect_tag, text_class_tag
       
    def __len__(self):
        return len(self.pairs)
