from datasets import load_dataset, Dataset, DatasetDict


def add_prefix(self, example):
    return self.prefix + example[self.source_lang]


def retrieve_relevant_validation_dataset(dataset_name, dataset_size):
    
    valid_dataset, valid_split = None, None
    if dataset_name == "wmt14":
        if dataset_size == "all":
            valid_split = "validation"
        else:
            if int(dataset_size) > 3000:
                print("You have selected a dataset size larger than the validation set. Using the entire validation set (3000).")
            dataset_size = min(int(dataset_size), 3000)
            valid_split = "validation[:{}]".format(dataset_size)
        
        valid_dataset = load_dataset(dataset_name, "fr-en", split=valid_split)
    elif dataset_name == "flores200" or dataset_name == "facebook/flores":
        if dataset_size == "all":
            valid_split = "dev"
        else:
            if int(dataset_size) > 997:
                print("You have selected a dataset size larger than the validation set. Using the entire validation set (997).")
            dataset_size = min(int(dataset_size), 997)
            valid_split = "dev[:{}]".format(dataset_size)
        
        dataset_flores200_fra = load_dataset("facebook/flores", "fra_Latn", split=valid_split)
        dataset_flores200_eng = load_dataset("facebook/flores", "eng_Latn", split=valid_split)

        translation_list = [{"en": dataset_flores200_eng['sentence'][idx], "fr": dataset_flores200_fra['sentence'][idx]} for idx in range(len(dataset_flores200_eng['sentence']))  ]
        
        dd_dict = {"translation": translation_list}
        valid_dataset = Dataset.from_dict(dd_dict)
        


        # valid_dataset = {"translation": {"en": dataset_flores200_eng['sentence'], "fr": dataset_flores200_fra['sentence']}}   
    else:
        raise NotImplementedError("Dataset not supported, or may be written incorrectly")
    
    return valid_dataset