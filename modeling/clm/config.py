from .argparser import get_args
_args = get_args()

if _args.base_model in ['xlnet-base-cased','transfo-xl-wt103']:
    ACCELERATOR = None
    GPUS = 1
    MAX_LENGTH = 2048
    MAX_CONTEXT_LENGTH = 1800
elif _args.base_model in ['google/reformer-crime-and-punishment']:
    ACCELERATOR = None
    GPUS = 1
    MAX_LENGTH = 524288
    MAX_CONTEXT_LENGTH = 20000
elif _args.base_model in ['google/reformer-enwik8']:
    ACCELERATOR = None
    GPUS = 1
    MAX_LENGTH = 65536
    MAX_CONTEXT_LENGTH = 20000
else:
    ACCELERATOR = 'dp'
    GPUS = -1
    MAX_LENGTH = 1024
    MAX_CONTEXT_LENGTH = 840

MAX_QUESTION_LENGTH = MAX_LENGTH - MAX_CONTEXT_LENGTH