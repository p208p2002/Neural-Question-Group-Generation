import argparse
from utils.argparser import create_general_parser

def get_args():
    parser = create_general_parser()
    args = parser.parse_args()
    return args