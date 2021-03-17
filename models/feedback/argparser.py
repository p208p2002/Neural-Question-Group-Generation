import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',default='facebook/bart-base',type=str, choices=['facebook/bart-base','facebook/bart-small'])
    parser.add_argument('--epoch',default=6,type=int)
    parser.add_argument('--batch_size',default=8,type=int)
    parser.add_argument('--lr',type=float,default=5e-5)
    parser.add_argument('--dev',type=int,default=0)
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,default=None)
    parser.add_argument('-ds','--datasets',nargs='+',default=['m_race'],required=False)
    parser.add_argument('-dnl','--disable_negative_loss',action='store_true')
    parser.add_argument('--alpha',type=float,default=1.0)
    args = parser.parse_args()

    allow_datasets = ['m_race']
    assert len(args.datasets)>0,'no datasets spec'
    for dataset in args.datasets:
        assert dataset in allow_datasets,'not allow dataset: `%s`'%dataset
    
    if 'm_race' in args.datasets: assert len(args.datasets) == 1,'m_race can use only alone'
        
    return args