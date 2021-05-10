from .argparser import get_args
_args = get_args()

ACCELERATOR = 'dp'
GPUS = _args.gpus
MAX_LENGTH = 512