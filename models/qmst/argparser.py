import argparse
from utils.argparser import create_base_parser

def get_args():
    parser = create_base_parser()
    parser.add_argument('-dnl','--disable_negative_loss',action='store_true')
    parser.add_argument('--alpha',type=float,default=0.02)
    parser.add_argument('--beta',type=float,default=0.965)
    args = parser.parse_args()
    return args