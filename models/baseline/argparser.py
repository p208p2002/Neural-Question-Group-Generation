import argparse
from utils.argparser import create_base_parser

def get_args():
    parser = create_base_parser()
    args = parser.parse_args()
    return args