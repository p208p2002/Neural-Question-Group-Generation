import json
import sys
from loguru import logger
import time
class Dataset():
    def __init__(self,model,path):
        self.model = model
        with open(path,'r',encoding='utf-8') as f:
            self.data = f.read().strip().split("\n")

    def __getitem__(self,index):
        return json.loads(self.data[index])
    
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    path_and_models = open(sys.argv[1],'r',encoding='utf-8').read().strip().split("\n")
    path_and_models = [tuple(path.split()) for path in path_and_models] #(model_tpye,file_path)

    current_index = 0
    total_model = len(path_and_models)

    datasets = []
    for (model,path) in path_and_models:
        datasets.append(Dataset(model,path))
    
    # make shure all dataset has same len
    data_len = len(datasets[0])
    for dataset in datasets:
        assert len(dataset) == data_len,f"{len(dataset)}, {data_len}"
    
    merge_outputs = []
    while(current_index < data_len):
        merge_output = {}
        article = datasets[0][current_index]['article']
        merge_output['_id'] = f"{current_index}_{article[:20]}"
        merge_output['_models'] = [d.model for d in datasets]
        
        optims = list(datasets[0][current_index].keys())
        optims.remove('article')
        
        merge_output['_optims'] = optims
        merge_output['article'] = article
        merge_output['questionGroups'] = []
        for dataset in datasets:
            _article = dataset[current_index]['article']
            assert article[:20] == _article[:20],'article is not equal'            
            for optim in optims:
                merge_output['questionGroups'].append(dataset[current_index][optim])
        merge_outputs.append(merge_output)
        current_index += 1
    
    # dump
    with open('./merge_model_output.json','w',encoding='utf-8') as f:
        f.write(json.dumps(merge_outputs))