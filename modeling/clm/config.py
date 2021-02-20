from .argparser import get_args
_args = get_args()

if _args.base_model in ['xlnet-base-cased']:
    MAX_LENGTH = 1600
    MAX_CONTEXT_LENGTH = 1200
else:
    MAX_LENGTH = 1024
    MAX_CONTEXT_LENGTH = 840

MAX_QUESTION_LENGTH = MAX_LENGTH - MAX_CONTEXT_LENGTH