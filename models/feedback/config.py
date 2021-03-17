from .argparser import get_args
from transformers import AutoConfig
_args = get_args()

MODEL_CONFIG = AutoConfig.from_pretrained(_args.base_model)

ACCELERATOR = 'dp'
GPUS = -1
MAX_LENGTH = 512
MAX_QUESTION_LENGTH = 32
WARN_UP_TOKEN = "^%"

