

# For measuring throughput, latency, and perhaps memoru footprint
speed_inference_metric_experiments = {   
    'batch_size': [1, 2, 4, 8, 16, 32, 64, 128],
    'dataset_size' : [1,100, 1000, "all"]
}


quality_experiment_configurations_sweep = {
    "dataset_size" : ["1","all"],
    "max_length" : [64,128,256,512], # Maximum generation sequence length
    "max_input_length": [10,20,30,50,None], # Set to None for no limit
    "beam_size" : [1,2,4,8,16,32,64],
    "model_name" : ["t5-small", "t5-base", "t5-large" , "facebook/nllb-200-distilled-600M", "facebook/nllb-200-1.3B"],
    "dataset_name" : ["wmt14", "flores200"],
    "tokenizer_padding_setting" : ["pad_to_max_length","no_pad_fill"],
    # "batch_size" : [1,32,64,128,256]
}



large_model_quality_experiments = {

}



# experiment_configurations_sweep_small = {
#     "dataset_size" : ["1","100", "1000", "all"],
#     "max_length" : [32,64,128,256,512], # Maximum sequence length
#     "beam_size" : [1,2,4],
#     "model_name" : ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b" , "nllb-200-distilled-600M"],
#     "dataset_name" : ["wmt14"],
#     # "batch_size" : ,
# }




# experiment_configurations_sweep_test = {
#     "dataset_size" : ["1","100"],
#     "max_length" : [16,512], # Maximum  sequence length
#     "model_name" : ["t5-small", "t5-base"],
# }
