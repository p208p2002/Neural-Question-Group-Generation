from modeling.clm.data_module import MergeRaceDataset


if __name__ == "__main__":
    train_datasets = [
        MergeRaceDataset('train','middle'),
        MergeRaceDataset('train','high'),
        MergeRaceDataset('dev','middle'),
        MergeRaceDataset('dev','high'),
        MergeRaceDataset('test','middle'),
        MergeRaceDataset('test','high')
    ]

    count_article = 0
    count_article_spec_question = 0
    count_general_question = 0

    for dataset in train_datasets:
        count_article += len(dataset)
        for _ in dataset:
            pass
        
        print('count_article',len(dataset))
        print('count_article_spec_question',dataset.count_article_spec_question)
        print('count_general_question',dataset.count_general_question)
        print()

        count_article_spec_question += dataset.count_article_spec_question
        count_general_question += dataset.count_general_question
    
    print('-'*100)
    print('count_article',count_article)
    print('count_article_spec_question',count_article_spec_question)
    print('count_general_question',count_general_question)
    
        