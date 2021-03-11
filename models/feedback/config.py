from .argparser import get_args
from transformers import AutoConfig
_args = get_args()

MODEL_CONFIG = AutoConfig.from_pretrained(_args.base_model)

ACCELERATOR = 'dp'
GPUS = -1
MAX_LENGTH = 1024

WARN_UP_TOKEN = "^%"

