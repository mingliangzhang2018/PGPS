import torch
import os
from PIL import Image
import datasets.diagram_aug as T_diagram
import datasets.text_aug as T_text
from datasets.operators import normalize_exp
from datasets.utils import get_combined_text, get_var_arg, get_text_index
from datasets.preprossing import SN

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, args, pairs, src_lang, tgt_lang, is_train=True):
        super().__init__()
        self.args = args
        self.pairs = pairs
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.is_train = is_train
        if is_train: 
            random_prob = args.random_prob
        else:
            random_prob = 0
        self.diagram_transform = T_diagram.Compose([
            T_diagram.Resize(args.diagram_size),
            T_diagram.CenterCrop(args.diagram_size),
            T_diagram.RandomFlip(random_prob),
            T_diagram.ToTensor(),
            T_diagram.Normalize()
        ])
        self.text_transform = T_text.Compose([
            T_text.Point_RandomReplace(random_prob),
            T_text.AngID_RandomReplace(random_prob),
            # T_text.Arg_RandomReplace(random_prob),
            T_text.StruPoint_RandomRotate(random_prob),
            # T_text.SemPoint_RandomRotate(random_prob),
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

        # diagram
        diagram_path = os.path.join(self.args.dataset_dir, 'Diagram', pair['diagram'])
        diagram = Image.open(diagram_path).convert("RGB")
        diagram = self.diagram_transform(diagram)
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
        # var and arg
        var_arg_positions, var_values, arg_values = \
                            get_var_arg(combine_text, self.args)
        # expression
        expression = normalize_exp(pair['expression'])
        expression = self.tgt_lang.indexes_from_sentence(expression, var_values, arg_values)
        # choices
        choices = [float(item) for item in pair['choices']]
        
        return  diagram, \
                text_token, text_sect_tag, text_class_tag, \
                var_arg_positions, var_values, arg_values, \
                expression, pair['answer'], pair['id'], choices 
                
    def __len__(self):
        return len(self.pairs)
