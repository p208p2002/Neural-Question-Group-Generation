import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',default='gpt2',type=str,dest='base_model')
    parser.add_argument('--epoch',default=10,type=int,dest='epoch')
    parser.add_argument('--batch_size',default=20,type=int,dest='batch_size')
    parser.add_argument('--lr',type=float,default=5e-5,dest='lr')
    parser.add_argument('--dev',action='store_true')
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,dest='from_checkpoint',default=None)
    args = parser.parse_args()
    return args