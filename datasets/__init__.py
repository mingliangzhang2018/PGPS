from torch.utils.data import DataLoader
from datasets.dataset import MyDataset
from torch.utils.data.distributed import DistributedSampler
from datasets.preprossing import *
import os

# def get_dataloader(args):

#     src_lang = SrcLang(args.vocab_src_path)
#     tgt_lang = TgtLang(args.vocab_tgt_path)

#     train_data_path = os.path.join(args.dataset_dir, args.dataset, 'train.json')
#     train_pairs = get_raw_pairs(train_data_path)
#     test_data_path = os.path.join(args.dataset_dir, args.dataset, 'test.json')
#     test_pairs = get_raw_pairs(test_data_path)

#     train_data = MyDataset(args, train_pairs, src_lang, tgt_lang, is_train=True)
#     train_sampler = DistributedSampler(train_data, shuffle=True)
#     train_loader = DataLoader(dataset=train_data, \
#                               batch_size=int(args.batch_size/args.nprocs), \
#                               pin_memory=True, \
#                               collate_fn=collater(args), \
#                               num_workers=args.workers, \
#                               sampler=train_sampler
#                               )
                              
#     test_data = MyDataset(args, test_pairs, src_lang, tgt_lang, is_train=False)
#     test_sampler = DistributedSampler(test_data, shuffle=False)
#     test_loader = DataLoader(dataset=test_data, \
#                             batch_size=1, \
#                             pin_memory=True, \
#                             collate_fn=collater(args), \
#                             num_workers=args.workers, \
#                             sampler=test_sampler
#                             )
                            
#     return train_loader, train_sampler, test_loader, src_lang, tgt_lang
from torch.utils.data import DataLoader, DistributedSampler
from transformers import T5Tokenizer

def get_dataloader(args):
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    src_lang = SrcLang(args.vocab_src_path)
    tgt_lang = TgtLang(args.vocab_tgt_path)

    train_data_path = os.path.join(args.dataset_dir, args.dataset, 'train.json')
    train_pairs = get_raw_pairs(train_data_path)
    test_data_path = os.path.join(args.dataset_dir, args.dataset, 'test.json')
    test_pairs = get_raw_pairs(test_data_path)

    train_pairs += test_pairs

    train_data = MyDataset(args, train_pairs, src_lang, tgt_lang, is_train=True)
    test_data = MyDataset(args, test_pairs, src_lang, tgt_lang, is_train=False)

    def collate_fn(batch):
        input_texts = [item['text'] for item in batch]
        target_texts = [item['target'] for item in batch]
        model_inputs = tokenizer(input_texts, max_length=args.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")
        labels = tokenizer(target_texts, max_length=args.max_seq_length, truncation=True, padding="max_length", return_tensors="pt").input_ids
        model_inputs["labels"] = labels
        return model_inputs

    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=int(args.batch_size / args.nprocs), 
                              pin_memory=True, 
                              collate_fn=collate_fn, 
                              num_workers=args.workers, 
                              sampler=train_sampler
                             )
    test_loader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             pin_memory=True, 
                             collate_fn=collate_fn, 
                             num_workers=args.workers, 
                             shuffle=False
                            )
                            
    return train_loader, train_sampler, test_loader, src_lang, tgt_lang
