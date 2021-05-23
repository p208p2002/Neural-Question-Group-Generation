
from functools import wraps
from time import time
import loguru
import os

def timer(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            loguru.logger.info(f"{func.__name__}, execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it


def save_huggingface_model(func):
    def wrapper(*_args,**_kwargs):
        this = _args[0]
        log_dir = os.path.join(this.trainer.default_root_dir,'dev') if this.trainer.log_dir is None else this.trainer.log_dir
        log_dir = os.path.join(log_dir,'huggingface_model')
        os.makedirs(log_dir,exist_ok=True)
        this.model.save_pretrained(log_dir)
        this.tokenizer.save_pretrained(log_dir)
        return func(*_args,**_kwargs)
    return wrapper