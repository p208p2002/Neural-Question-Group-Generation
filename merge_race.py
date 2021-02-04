import json
import os
from torch.utils.data import Dataset

class RaceDataset(Dataset):
    def __init__(self,split_set,level,dataset_dir='datasets/RACE',no_label=False):
        super().__init__()
        assert split_set in ['dev','test','train']
        assert level in ['all','middle','high']
        self.all_file_paths = []
        for root, dirs, files in os.walk(os.path.join(dataset_dir,split_set)):
            for f in files:
                if level == 'all':
                    self.all_file_paths.append(os.path.join(root,f))
                elif root == os.path.join(dataset_dir,split_set,level):
                    self.all_file_paths.append(os.path.join(root,f))
            
    def __getitem__(self,index):
        with open(self.all_file_paths[index],'r',encoding='utf-8') as f:
            data = json.load(f)
            _questions = data['questions']

            # keep only type is question
            questions = []
            for _q in _questions:
                if _q[-1] == '?': 
                    questions.append(_q)
            
            data['questions'] = questions
            return data
            
    def __len__(self):
        return len(self.all_file_paths)

if __name__ == "__main__":
    splits = ['dev','test','train']
    levels = ['high','middle']

    output_dir = 'datasets/merge-race'
    os.system('rm -rf %s'%output_dir)
    os.makedirs(output_dir,exist_ok=True)

    match_count = 0
    step = 0
    total = 0

    for split in splits:
        with open(os.path.join('datasets/EQG-RACE',split+'.json'),'r',encoding='utf-8') as f:
            eqg_race = json.load(f)
            total += len(eqg_race)
            for eqg_race_data in eqg_race: # dict_keys(['question', 'max_sent', 'tag', 'rouge', 'answer', 'sent'])
                eqg_race_question = eqg_race_data['question'].lower().replace(" ","")[8:-1]
                
                race_datasets = []
                for level in levels:
                    race_dataset = RaceDataset(split,level)
                    race_datasets.append(race_dataset)

                for li,level in enumerate(levels):
                    race_dataset = race_datasets[li]
                    merge_race_dir = os.path.join(output_dir,split,level)
                    os.makedirs(merge_race_dir,exist_ok=True)

                    with open(os.path.join(merge_race_dir,level+'.jsonl'),'a') as merge_race_f:
                        for race_data in race_dataset:
                            race_questions = race_data['questions']
                            race_data['acticle_spec_questions'] = []
                            for race_question in race_questions:
                                _race_question = race_question.lower().replace(" ","")[8:-1]
                                if _race_question == eqg_race_question:
                                    race_data['acticle_spec_questions'].append(race_question)
                                    merge_race_f.write(json.dumps(race_data)+'\n')
                                    match_count += 1
                                step+=1
                                if step%3000==0:
                                    print('match_count(miss_match): %d/%d(%d)'%(match_count,total,total-match_count),'step:',str(step/1000)+'k',"{:<50}".format(merge_race_f.name),end='\r')
    print('match_count(miss_match): %d/%d(%d)'%(match_count,total,total-match_count),'step:',str(step/1000)+'k',"{:<50}".format(merge_race_f.name),end='\n')