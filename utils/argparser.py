import argparse
import sys
from loguru import logger

def create_base_parser():
    """
    the general parser for all models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',default='facebook/bart-base',type=str, choices=['t5-small','t5-base','facebook/bart-base','facebook/bart-small'])
    parser.add_argument('--gen_n',default=10,type=int)
    parser.add_argument('--pick_n',default=5,type=int)
    parser.add_argument('-g_opts','--qgg_optims',default=['ga','first-n','random','greedy'], choices=['ga','first-n','random','greedy'], nargs='+', required=False)
    parser.add_argument('--epoch',default=10,type=int)
    parser.add_argument('--batch_size',default=4,type=int)
    parser.add_argument('--lr',type=float,default=5e-6)
    parser.add_argument('--dev',type=int,default=0)
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,default=None)
    parser.add_argument('--use_subsets', default=['s-type','c-type'], choices=['s-type','c-type','g-type'], nargs='+', required=False)
    parser.add_argument('--gen_target',default='only-q',choices=['only-q','q-and-a'], required=False)
    parser.add_argument('-m','--message', required=True)
    parser.add_argument('--_argv',default=' '.join(sys.argv))
    parser.add_argument('-ghed','--gen_human_eval_data',action='store_true')

    # 
    args, unknown = parser.parse_known_args()
    if args.gen_target == 'q-and-a':
        logger.warning("`args.gen_target == 'q-and-a'` is deprecated")
        raise Exception()
    return parser

def get_general_args():
    parser = create_base_parser()
    args, unknown = parser.parse_known_args()
    return args