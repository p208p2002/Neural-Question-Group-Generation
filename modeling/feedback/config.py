from .argparser import get_args
_args = get_args()

if _args.base_model in ['xlnet-base-cased','transfo-xl-wt103']:
    assert False
elif _args.base_model in ['google/reformer-crime-and-punishment']:
    assert False
elif _args.base_model in ['google/reformer-enwik8']:
    assert False
else:
    ACCELERATOR = 'dp'
    GPUS = -1
    MAX_LENGTH = 1024