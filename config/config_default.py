import argparse
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
                   
criterion_list = ["CrossEntropy", "FocalLoss", "MaskedCrossEntropy", "MLMCrossEntropy"]
optimizer_list = ["SGD", "ADAM"]
scheduler_list = ["multistep",'cosine','warmup']
visual_backbone_list = ['ResNet10', 'mobilenet_v2']
encoder_list = ['lstm', 'gru', 'transformer']
decoder_list = ["rnn_decoder", "tree_decoder"]
dataset_list = ['Geometry3K', 'PGPS9K'] 


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch PGPS Training')
    # visual backbone
    ##############################################################################
    parser.add_argument('--visual_backbone', default="ResNet10", type=str, choices=visual_backbone_list)
    parser.add_argument('--diagram_size',  default=128, type=int)
    # encoder model
    ##############################################################################
    parser.add_argument('--encoder_type', default="gru", type=str, choices=encoder_list)
    parser.add_argument('--encoder_layers', default=2, type=int)
    parser.add_argument('--encoder_embedding_size', default=256, type=int)
    parser.add_argument('--encoder_hidden_size', default=256, type=int)
    parser.add_argument('--max_input_len', default=400, type=int)
    # decoder model
    ##############################################################################
    parser.add_argument('--decoder_type', default="rnn_decoder", type=str, choices=decoder_list)
    parser.add_argument('--decoder_layers', default=2, type=int)
    parser.add_argument('--decoder_embedding_size', default=256, type=int)
    parser.add_argument('--decoder_hidden_size', default=256, type=int)
    parser.add_argument('--max_output_len', default=40, type=int)
    # general model
    ##############################################################################
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--beam_size', default=10, type=int)
    # optimizer
    ##############################################################################
    parser.add_argument('--optimizer_type', default="ADAMW", type=str, choices=optimizer_list)
    parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--max_epoch', default=4000, type=int)
    parser.add_argument('--scheduler_type', default="warmup", type=str, choices=scheduler_list)
    parser.add_argument('--scheduler_step', default=[1000, 2000, 3000], type=list)
    parser.add_argument('--scheduler_factor', default=0.4, type=float, help='learning rate decay factor')
    parser.add_argument('--cosine_decay_end', default=0.0, type=float, help='cosine decay end')
    parser.add_argument('--warm_epoch', default=40, type=int)
    # criterion
    ###############################################################################
    parser.add_argument('--criterion', default="MLMCrossEntropy", choices=criterion_list, type=str)
    # dataset      
    #################################################################################
    parser.add_argument('--dataset', default="PGPS9K", type=str, choices=dataset_list)
    parser.add_argument('--dataset_dir', default='/lustre/home/mlzhang/Datasets/PGPS9K_all')
    parser.add_argument('--pretrain_vis_path', default='')
    parser.add_argument('--vocab_src_path', default='./vocab/vocab_src.txt')
    parser.add_argument('--vocab_tgt_path', default='./vocab/vocab_tgt.txt')
    parser.add_argument('--pretrain_emb_path', default='')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--random_prob', default=0.7, type=float)
    parser.add_argument('--without_stru', action='store_true', help='structure clauses are used or not')
    parser.add_argument('--trim_min_count', default=5, type=int, help='minimum number of word')
    parser.add_argument('--mlm_probability', default=0.3, type=float)
    # print information
    ###################################################################################
    parser.add_argument('--dump_path', default="./log/", type=str, help='save log path')
    parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
    parser.add_argument('--eval_epoch', default=500, type=int)
    # general config
    ###################################################################################
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--resume_model', default="", type=str, help='use pre-trained model')
    # DistributedDataParallel
    ###################################################################################
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--init_method', default="env://", type=str, help='distributed init method')
    parser.add_argument('--debug', action='store_true', help = "if debug than set local rank = 0")
    parser.add_argument('--seed', default=202302, type=int,help='seed for initializing training. ')

    return parser.parse_args()
