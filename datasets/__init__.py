from torch.utils.data import DataLoader
from datasets.dataset import MyDataset
from torch.utils.data.distributed import DistributedSampler
from datasets.preprossing import *
import os

def get_dataloader(args):

    src_lang = SrcLang(args.vocab_src_path)
    tgt_lang = TgtLang(args.vocab_tgt_path)

    train_data_path = os.path.join(args.dataset_dir, args.dataset, 'train.json')
    train_pairs = get_raw_pairs(train_data_path)
    test_data_path = os.path.join(args.dataset_dir, args.dataset, 'test.json')
    test_pairs = get_raw_pairs(test_data_path)

    train_data = MyDataset(args, train_pairs, src_lang, tgt_lang, is_train=True)
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(dataset=train_data, \
                              batch_size=int(args.batch_size/args.nprocs), \
                              pin_memory=True, \
                              collate_fn=collater(args), \
                              num_workers=args.workers, \
                              sampler=train_sampler
                              )
                              
    test_data = MyDataset(args, test_pairs, src_lang, tgt_lang, is_train=False)
    test_sampler = DistributedSampler(test_data, shuffle=False)
    test_loader = DataLoader(dataset=test_data, \
                            batch_size=1, \
                            pin_memory=True, \
                            collate_fn=collater(args), \
                            num_workers=args.workers, \
                            sampler=test_sampler
                            )
                            
    return train_loader, train_sampler, test_loader, src_lang, tgt_lang
