from .argparser import get_args
_args = get_args()


ACCELERATOR = 'dp'
GPUS = -1
MAX_LENGTH = 1024
SEP_TOKEN = '[SEP]'