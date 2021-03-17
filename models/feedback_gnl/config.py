from .argparser import get_args
from transformers import AutoConfig
_args = get_args()

MODEL_CONFIG = AutoConfig.from_pretrained(_args.base_model)

ACCELERATOR = None
GPUS = 1
MAX_LENGTH = 512