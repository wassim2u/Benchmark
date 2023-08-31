

dataset_name_map = {
    "facebook/flores" : "flores200"
}

dataset_info = {
    "wmt14": {
        "en-fr": "fr-en"
    },
    "opus100": {
        "en-fr": "en-fr"
    },
    "flores200":{
        
    }
}

lang_code_t5_dictionary = {
    "en": "English",
    "fr" : "French",
}


def get_dataset_info(dataset_name, dataset_size, source_lang, target_lang):
    dataset_size = int(dataset_size)
    dataset_name = dataset_name_map.get(dataset_name, dataset_name)
    dataset_info_dict = dataset_info[dataset_name]
    dataset_info_dict = dataset_info_dict[source_lang + "-" + target_lang]
    return dataset_name, dataset_size, dataset_info_dict
