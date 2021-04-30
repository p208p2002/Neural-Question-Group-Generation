import os
import json

def setup_logger(func):
    def wrapper(*args,**kwargs):
        self = args[0]
        self._log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        self.predict_logger = PredictLogger(save_dir=self._log_dir)
        return func(*args,**kwargs)
    return wrapper

class PredictLogger():
    def __init__(self, save_dir):
        save_dir = os.path.join(save_dir,'predicts')
        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir
        self._index = 0
        self.count = 0

    def log(self, log_dict):
        with open(os.path.join(self.save_dir,'log.jsonl'),'a',encoding='utf-8') as f:
            f.write(json.dumps(log_dict)+"\n")

        with open(os.path.join(self.save_dir,'%d.txt'%self._index),'a',encoding='utf-8') as f:
            for key,item in log_dict.items():
                f.write('-'*10+" "+key+" %d "%self.count+'-'*10+'\n')
                if type(item) == list:
                    item = [str(i)+"\n" for i in item]
                    item = "- " + "- ".join(item)
                f.write(str(item))
                f.write("\n")
                f.write("\n")
        self.count +=1
        if self.count % 100 == 0:
            self._index += 1