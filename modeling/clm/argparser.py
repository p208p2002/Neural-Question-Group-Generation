import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',default='gpt2',type=str)
    parser.add_argument('--epoch',default=10,type=int)
    parser.add_argument('--batch_size',default=10,type=int)
    parser.add_argument('--lr',type=float,default=5e-5)
    parser.add_argument('--dev',type=int,default=0)
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,default=None)
    parser.add_argument('-d','--dataset',choices=['race','g_race','eqg',],required=True)
    args = parser.parse_args()
    return args