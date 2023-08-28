

# For measuring throughput, latency, and perhaps memoru footprint
speed_inference_metric_experiments = {   
    'batch_size': [1, 2, 4, 8, 16, 32, 64, 128],
    'dataset_size' : [1,100, 1000, "all"]
}


# quality_experiment_configurations_sweep = {
#     "dataset_size" : ["all", 1],
#     "max_gen_length" : [64,128,256,512], # Maximum generation sequence length
#     "max_input_length": [10,30,-1], # Set to -1 for no limit
#     "beam_size" : [1,2,4,8,16],
#     "model_name" : ["t5-base", "t5-large", "facebook/nllb-200-distilled-600M"],
#     "dataset_name" : ["wmt14", "flores200"],
#     "tokenizer_padding_setting" : ["pad_to_max_length","do_not_pad"],
#     # "batch_size" : [1,32,64,128,256]
# }



quality_experiment_configurations_sweep = {
    "dataset_size" : ["all", 1],
    "max_gen_length" : [128], # Maximum generation sequence length
    "max_input_length": [-1,15,30], # Set to -1 for no limit
    "beam_size" : [1,2,4,8],
    "model_name" : ["facebook/nllb-200-distilled-600M", "t5-large", "t5-base"],
    "dataset_name" : ["wmt14", "flores200"],
    "tokenizer_padding_setting" : ["pad_to_max_length","do_not_pad"],
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
