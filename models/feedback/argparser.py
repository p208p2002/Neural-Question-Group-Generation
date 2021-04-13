import argparse
from utils.argparser import create_general_parser

def get_args():
    parser = create_general_parser()
    parser.add_argument('-dnl','--disable_negative_loss',action='store_true')
    parser.add_argument('--alpha',type=float,default=0.05)
    parser.add_argument('--beta',type=float,default=0.75)
    args = parser.parse_args()
    return args