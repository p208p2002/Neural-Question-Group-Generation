import argparse

def create_general_parser():
    """
    the general parser for all models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',default='facebook/bart-base',type=str, choices=['t5-small','t5-base','facebook/bart-base','facebook/bart-small'])
    parser.add_argument('--gen_n',default=10,type=int)
    parser.add_argument('--pick_n',default=5,type=int)
    parser.add_argument('--qgg_optim',default='ga',type=str, choices=['ga','first-n','random','greedy'])
    parser.add_argument('--epoch',default=10,type=int)
    parser.add_argument('--batch_size',default=8,type=int)
    parser.add_argument('--lr',type=float,default=5e-6)
    parser.add_argument('--dev',type=int,default=0)
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,default=None)
    parser.add_argument('-ds','--datasets',nargs='+',default=['m_race'],required=False)

    return parser