import csv
from typing import Any, Callable
import functools

import torch
from transformers import  AutoConfig, NllbTokenizer, AutoTokenizer, T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration, M2M100ForConditionalGeneration, NllbMoeForConditionalGeneration, SwitchTransformersForConditionalGeneration
from torch.profiler import profile, record_function, ProfilerActivity

from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import IterableDataset, Subset, DataLoader
import pandas as pd
import numpy as np
import evaluate
import os
from archer.runtime import ArcherEngine
import getpass
from transformers.models.t5.modeling_t5 import (T5Stack, T5EncoderModel)
from transformers.models.nllb_moe.modeling_nllb_moe import (
    NllbMoeEncoder,
    NllbMoeDecoder,
)
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersStack, )

from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100Decoder, M2M100Encoder
)
import transformers


# from deepspeed.accelerator import get_accelerator
# from deepspeed.profiling.flops_profiler import get_model_profile
import utils
from hyperparameters_config import quality_experiment_configurations_sweep, large_model_quality_experiments

from tqdm.auto import tqdm
import time
import argparse
import traceback     
                
                
import pathlib

def setup_environment():
    num_gpus = torch.cuda.device_count()
    for gid in range(num_gpus):
        torch.cuda.set_device(gid)
        if torch.cuda.utilization() == 0:
            break
setup_environment()


CONFIG = {"nvme_path": f"/mnt/{getpass.getuser()}/test-data"}
METRIC_REPORTS_DIR_PATH = "./metric_logs/"

FORWARD_TIMES_FILENAMES = {
    "t5-small" : METRIC_REPORTS_DIR_PATH + "t5_latencies.csv",
    "t5-base" : METRIC_REPORTS_DIR_PATH + "t5_latencies.csv",
    "t5-large": METRIC_REPORTS_DIR_PATH + "t5_latencies.csv",
    "nllb-200-1.3B" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "nllb-200-distilled-600M" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "nllb-moe-54b" : METRIC_REPORTS_DIR_PATH + "nllb-moe_latencies.csv",
    "google/switch-base-128" : METRIC_REPORTS_DIR_PATH + "switch_latencies.csv",
    "facebook/nllb-200-distilled-1.3B" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "facebook/nllb-200-distilled-600M" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "facebook/nllb-200-1.3B" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
}

def add_latency_measurement_functionality():
  
    transformers.models.t5.modeling_t5._old_t5_forward = T5Stack.forward
    transformers.models.nllb_moe.modeling_nllb_moe._old_nllb_moe_encoder_forward = (
        NllbMoeEncoder.forward)
    transformers.models.nllb_moe.modeling_nllb_moe._old_nllb_moe_decoder_forward = (
        NllbMoeDecoder.forward)
    transformers.models.switch_transformers._old_switch_transformers_forward = (
        SwitchTransformersStack.forward)
    transformers.models.m2m_100.modeling_m2m_100._old_m2m_100_encoder_forward = (
        M2M100Encoder.forward)
    transformers.models.m2m_100.modeling_m2m_100._old_m2m_100_decoder_forward = (
        M2M100Decoder.forward)



    T5Stack.forward = forward_decorator(T5Stack.forward, METRIC_REPORTS_DIR_PATH + "t5_latencies.csv")
    NllbMoeEncoder.forward = forward_decorator(NllbMoeEncoder.forward, METRIC_REPORTS_DIR_PATH + "nllb-moe_latencies.csv")
    NllbMoeDecoder.forward = forward_decorator(NllbMoeDecoder.forward,  METRIC_REPORTS_DIR_PATH + "nllb-moe_latencies.csv")
    SwitchTransformersStack.forward = forward_decorator(
        SwitchTransformersStack.forward, METRIC_REPORTS_DIR_PATH + "switch_latencies.csv")
    M2M100Encoder.forward = forward_decorator(M2M100Encoder.forward, METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv")
    M2M100Decoder.forward = forward_decorator(M2M100Decoder.forward, METRIC_REPORTS_DIR_PATH +  "nllb_latencies.csv")
    

def forward_decorator(orig_forward: Callable, path_to_logfile) -> Callable:

    @functools.wraps(orig_forward)
    def timer_forward(cls, *args, **kwargs):
        start_time = time.perf_counter()
        result = orig_forward(cls, *args, **kwargs)
        end_time = time.perf_counter()

        # cls.layer_latencies = {"encoder": None, "decoder": []} #Take average for decoder, and for encoder you should take latency per token (divide by sequence length)
        if hasattr(cls, "is_decoder"):
            layer_type = "decoder" if cls.is_decoder else "encoder"
            
        else:
            layer_type = "encoder" if isinstance(cls,
                                                 NllbMoeEncoder) or isinstance(cls,M2M100Encoder) or isinstance(cls, T5EncoderModel) else "decoder"
            
        
        with open(path_to_logfile, 'a') as f:
            # f.write(f"{layer_type},{end_time - start_time}\n")

            writer = csv.writer(f)
            writer.writerow([layer_type, end_time - start_time])
            


        return result

    return timer_forward

# Add functionality to original transformers forward method to log latency
add_latency_measurement_functionality()


source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "


# clf_metrics = evaluate.combine(["sacrebleu", "f1", "precision", "recall"])

sacrebleu = evaluate.load("sacrebleu") # Sacrebleu with moses tokenizer
spBleu = evaluate.load("sacrebleu") # spBleu: Sacrebleu with sentencepiece (FLORES-200 Tokenizer)
chrf= evaluate.load("chrf") #chrf : Word order = 0 
chrfpp = evaluate.load("chrf") #chrf++ : Word order = 2 

metrics_name_map_dict = {
    "sacrebleu" : "sacrebleu",
    "spBleu" : "sacrebleu",
    "chrf" : "chrf",
    "chrf++" : "chrf",
    "chrfpp" : "chrf",
    "comet": None,
    "rouge": None,
    "meteor" : "meteor",
}
 
# clf_metrics = evaluate.combine(["sacrebleu", "sacrebleu", "chrf", "chrf", "comet", "rouge", "meteor",])


# max_length = 128

# max_source_length = 1024

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
    "fr" : "French"
}

# models = ["t5-small", "t5-base", "t5-large", "google/switch-base-128", "nllb-200-distilled-600M", "nllb-moe-54b"]



# We define this postprocess() function that takes predictions and labels and converts them to the lists of strings our metric object will expect:
def postprocess_batch(predictions, labels, tokenizer):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    # print(decoded_preds)
    # print(decoded_labels)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    
    return decoded_preds, decoded_labels, prediction_lens


def postprocess_text(preds, labels):
    postprocessed_preds = [pred.strip() for pred in preds]
    postprocessed_labels = [[label.strip()] for label in labels]

    return postprocessed_preds, postprocessed_labels





# def compute_metrics(metric, eval_preds):
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["score"]}

#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     return result




class Evaluation:
    def __init__(self,model_name, evaluation_dataset, dataset_name, metrics_args, hyperparameters={}, streaming=False, src_lang="en", tgt_lang="fr"):
        
        self.model_name = model_name
        self.evaluation_dataset = evaluation_dataset
        self.dataset_name = dataset_name
        # Convert troublesome naming just in case to make creating filenames easier. If passing facebook/flores, convert to flores200 for instance
        if dataset_name in dataset_name_map.keys():
            self.dataset_name = dataset_name_map[dataset_name]
        
        
        self.streaming = streaming
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if src_lang == "en" and tgt_lang == "fr" and "t5" in model_name or "google/switch" in model_name:
            self.prefix = "translate English to French: "
        else:
            self.prefix = ""

        self.hyperparameters = hyperparameters
        # Define the metrics
        self.all_metrics = self.load_relevant_metrics(metrics_args)

        # Add functionality to original transformers forward method to log latency
        # self.PATH_TO_LOG_LATENCY = METRIC_REPORTS_DIR_PATH + self.model_name + "_"+ self.dataset_name + "_latencies.csv"
        # # TODO: Hacky way to remove these attributes from file path. Make the log name conversion more robust
        # self.PATH_TO_LOG_LATENCY = self.PATH_TO_LOG_LATENCY.replace("facebook/", "")
        # self.PATH_TO_LOG_LATENCY = self.PATH_TO_LOG_LATENCY.replace("google/", "")
        # self.add_latency_measurement_functionality( path_to_log_latency =  self.PATH_TO_LOG_LATENCY)
        
  

        
        # Define model and tokenizer
        if model_name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/t5-v1_1-small"]: 
            # self.tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=max_length)
            # self.tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=512)
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        elif model_name in ["facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-1.3B"] and src_lang == "en" and tgt_lang == "fr":
            self.tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="fra_Latn")
            self.model= M2M100ForConditionalGeneration.from_pretrained(model_name)


        elif "nllb-moe-54b" in model_name and src_lang == "en" and tgt_lang == "fr":
            CONFIG["nvme_path"] = os.path.join(CONFIG["nvme_path"], "nllb-moe-54b")
            # self.tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="fra_Latn")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="fra_Latn")

            # Model would be too huge to load on GPU. Offload to Disk using Archer
            self.archer_engine = ArcherEngine()
            with self.archer_engine.init(NllbMoeForConditionalGeneration,
                            ar_config=CONFIG,
                            trace_path="./trace.json"):
                # model_offload = NllbMoeModel.from_pretrained(model_name)
                self.model = NllbMoeForConditionalGeneration.from_pretrained(model_name)
            

        elif "google/switch-base-128" in model_name:
            CONFIG["nvme_path"] = os.path.join(CONFIG["nvme_path"], "switch-base-128")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Model would be too huge to load on GPU. Offload to Disk using Archer
            self.archer_engine = ArcherEngine()
            with self.archer_engine.init(SwitchTransformersForConditionalGeneration,
                                    ar_config=CONFIG,
                                    trace_path="./trace.json"):
                self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
                    'google/switch-base-128')

        else:
            raise NotImplementedError("Model not supported")


        self.model.eval()
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: Check if this is correct
        if self.archer_engine is None:
            self.model.to(self.device)
        



    def set_other_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

        
    # def __call__(self, deepspeed=False, save_profile=False, export_trace_file=False, export_flame_graph=False, *args: Any, **kwds: Any) -> Any:
    #     if deepspeed:
    #         with get_accelerator().device(0):
    #             flops, macs, params = get_model_profile(
    #                 self.model,
    #                 kwargs=self.preprocess_function_constructor(),
    #                 print_profile=True,
    #                 detailed=True,
    #             )
    #             print(flops)
    #             print(macs)
    #             print(params)
    #             exit()

                


    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as profInference:
    #         # TODO: torch.no_grad() or without it?
    #         with record_function("model_inference"):
    #             # TODO: Use datsets.map() instead of this function
    #             self.inputs = self.preprocess_function().input_ids.to(self.device)
    #             with torch.no_grad():
    #                 outputs = self.model.generate(self.inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    #             decoded_labels = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        
    #     result = self.calculate_bleu(decoded_labels, self.evaluation_dataset["translation"])
    #     print(result)
        
        
    #     # Get the resource utilization from the profiler 

    #     res_util_df = pd.DataFrame({e.key:e.__dict__ for e in profInference.key_averages()}).T
    #     res_util_df[['count', 'cpu_time_total', 'cuda_time_total']].sort_values(['cuda_time_total', 'cpu_time_total'], ascending=False)
    #     if save_profile:
    #         res_util_df.to_csv(METRIC_REPORTS_DIR_PATH + self.model_name + "_"+ self.dataset_name + "_res_util_full.csv")
    #     print(res_util_df.head(10))
    
    #     # # Calculate Throughput:  Get the total time and number of examples from the profiler
    #     # print(profInference.self_cpu_time_total)
    #     # total_time = float(profInference.self_cpu_time_total) / 1000.0  # Convert to seconds
    #     # print(total_time)
    #     # total_examples = len(self.evaluation_dataset["translation"])


  
    def call_batched_w_profiler():
        pass

    # Returns False if execution should be skipped or Failed
    def call_batched(self, beam_size=1, 
                    max_gen_length=128,
                    max_input_seq_length=128,
                    batch_size=128, 
                    pytorch_profiling=False, save_res_util=False, save_metrics_csv=False, overwrite_csv=False):
        from transformers import DataCollatorForSeq2Seq
        # If path already exists for recording latencies using decorated function, it is deleted. This is to avoid rewrapping the model twice.
        self.PATH_TO_LOG_LATENCY = FORWARD_TIMES_FILENAMES[self.model_name]
        if pathlib.Path(self.PATH_TO_LOG_LATENCY).exists():
            os.remove(self.PATH_TO_LOG_LATENCY)
        
        filtered_dataset = None
        if self.hyperparameters.get("max_input_seq_length") is not None and self.hyperparameters["max_input_seq_length"] != -1:
            print("Filtering dataset by max_input_seq_length")
            filtered_dataset= self.evaluation_dataset.filter(lambda example: (len(example["translation"]["en"].split(" "))) <= self.hyperparameters["max_input_seq_length"])

            if len(filtered_dataset) == 0:
                print("Skip this experiment because dataset is empty after filtering by max_input_seq_length")
                return False
        else:
            filtered_dataset = self.evaluation_dataset
        print(len(filtered_dataset))

        tokenized_datasets = filtered_dataset.map(
            self.preprocess_function_with_text_target,
            batched=True,
            remove_columns=filtered_dataset.column_names,
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        # TODO: should i add label_pad_token_id??
        eval_dataloader = DataLoader(
                tokenized_datasets, collate_fn=data_collator, batch_size=batch_size, pin_memory=True, 
        )

        print("what")
        pred_lengths = np.array([])
        # Call config to retrieve pad_token_id. Will be useful for calculating throughput.
        config = AutoConfig.from_pretrained(self.model_name)
        self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
        number_of_total_encoded_tokens = 0
        number_of_total_decoded_tokens = 0
        total_encoded_tokens = []


        if not pytorch_profiling:
            
            for batch in tqdm(eval_dataloader):
                    decoder_start_token_id = self.model.config.pad_token_id

                    input_ids = batch["input_ids"].to(self.device, non_blocking=True)

                    attention_mask = (input_ids != config.pad_token_id).to(input_ids.dtype)

                    with torch.no_grad():
                        if  "nllb" in self.model_name:
                            outputs = self.model.generate(input_ids, forced_bos_token_id=self.tokenizer.lang_code_to_id["fra_Latn"], do_sample=False, max_length=max_gen_length, num_beams=beam_size)
                        elif "switch" in self.model_name:
                            print("here!")
                            print(self.model.device)
                            print(input_ids.device)
                            outputs = self.model.generate(input_ids, do_sample=False, max_length=max_gen_length, num_beams=beam_size,   decoder_start_token_id=decoder_start_token_id)

                        else:
                            print("no way")
                            # outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=128, num_beams=1) # Works, for transformers.
                            outputs = self.model.generate(input_ids, do_sample=False, max_length=max_gen_length, num_beams=beam_size)

                    del input_ids
                    decoded_preds, decoded_labels, pred_length  = postprocess_batch(outputs, batch["labels"], self.tokenizer)
                    del outputs

                    for metric_name in self.all_metrics.keys():
                        self.all_metrics[metric_name].add_batch(predictions=decoded_preds, references=decoded_labels)

                    pred_lengths = np.concatenate((pred_lengths, pred_length), axis=0) if len(pred_lengths) !=0 else pred_length
                
                    decoded_preds, decoded_labels = None, None

                    # For calculating throughput, we need to know the number of total tokens generated.
                    number_of_total_encoded_tokens += torch.sum(attention_mask)
                    number_of_total_decoded_tokens += np.sum(pred_length) #Sum of generated lengths
                    total_encoded_tokens.append(torch.sum(attention_mask))  #Number of tokens. Will be used to calculate latency per token



                    del attention_mask
                    import gc
                    gc.collect()
                    # torch.cuda.synchronize()
                    torch.cuda.empty_cache()


        else:
            pass
          

        # Export to 
        # profInference.export_chrome_trace("trace.json")

        print("Generation done!")
        # Retrieve and process latency from log file
        # forward_times_df = pd.read_csv(self.PATH_TO_LOG_LATENCY, names=["layer_type", "latency_s"]).groupby("layer_type").mean() 
                
        forward_times_df = pd.read_csv(self.PATH_TO_LOG_LATENCY, names=["layer_type", "latency_s"])
        forward_times_df.reset_index(drop=True,inplace=True)
        encoder_df = forward_times_df[forward_times_df['layer_type'] == "encoder"]
        for i in range(len(total_encoded_tokens)):
            # Calculate latency per token! Normalise the results by dividing by the number of tokens
            encoder_df.iloc[i]["latency_s"] = (encoder_df.iloc[i]["latency_s"]) / (total_encoded_tokens[i]).cpu().numpy()
        try:
            # TODO: Potential bug: Check why this fails for t5-small
            # assert(len(total_encoded_tokens) == len(encoder_df))
            if (len(total_encoded_tokens) != len(encoder_df)):
                print("WARNING! Number of total encoded tokens and number of encoder forward passes are not equal.")

                with open("./fail_logs/{}.txt".format(self.model_name.replace("facebook/","").replace("google/","")), "a") as log:
                    log.write("WARNING! Number of total encoded tokens and number of encoder forward passes are not equal.")
                    text = f"Hyperparameters: model_name: {self.model_name}, Batch_size: {batch_size}, dataset_size: {len(self.evaluation_dataset)}, dataset_size_after_filter_wrt_input_len: {len(filtered_dataset)} max_gen_length: {max_gen_length}, beam_size: {beam_size}, max_input_seq_length: {self.hyperparameters['max_input_seq_length']}, tokenizer_padding_setting: {self.hyperparameters['tokenizer_padding_setting']}, dataset_name= {self.dataset_name} \n"
                    log.write(text)
        except AssertionError as e:
            print(e)
            print(self.model_name)
            print(self.hyperparameters)
            print(len(total_encoded_tokens))
            print(len(encoder_df))
            print(encoder_df)
            
            raise AssertionError("Number of total encoded tokens and number of encoder forward passes are not equal. Something went wrong.")

        
        forward_times_df[forward_times_df['layer_type'] == "encoder"] = encoder_df
        latencies_df = forward_times_df.groupby("layer_type").mean()
        
        # Calculate latency

        encoder_latency = latencies_df.loc["encoder"]["latency_s"]
        decoder_latency = latencies_df.loc["decoder"]["latency_s"]
        print(f"Encoder latency (per token, averaged across all samples): {encoder_latency}")
        print(f"Decoder latency (averaged): {decoder_latency}")

        # Calculate throughput
        encoder_throughput = (number_of_total_encoded_tokens/ encoder_latency).cpu().numpy()
        decoder_throughput = (number_of_total_decoded_tokens/ decoder_latency)
        print(f"Encoder throughput: {encoder_throughput}")
        print(f"Decoder throughput: {decoder_throughput}")
        # Delete the log file
        os.remove(self.PATH_TO_LOG_LATENCY)
        # Calculate Quality Metrics
        metric_results_dict = self.report_metrics(pred_lengths)

        if save_metrics_csv:
            metric_results_dict_w_params = {}
            metric_results_dict_w_params["model_name"] = self.model_name
            total_params = sum(p.numel() for p in self.model.parameters())
            metric_results_dict_w_params["total_params"] = total_params
            metric_results_dict_w_params["dataset_name"] = self.dataset_name
            metric_results_dict_w_params["dataset_size"] = len(self.evaluation_dataset)
            metric_results_dict_w_params["dataset_size_after_filter_wrt_input_len"] = len(filtered_dataset)
            metric_results_dict_w_params["src_lang"] = self.src_lang
            metric_results_dict_w_params["tgt_lang"] = self.tgt_lang
            metric_results_dict_w_params["batch_size"] = batch_size
            metric_results_dict_w_params["beam_size"] = beam_size
            metric_results_dict_w_params["max_gen_length"] = max_gen_length
            metric_results_dict_w_params["max_input_seq_length"] = self.hyperparameters["max_input_seq_length"] # TODO: Report on different sentence lengths.
            metric_results_dict_w_params["tokenizer_padding_setting"] = self.hyperparameters["tokenizer_padding_setting"] # TODO: Report on different sentence lengths.

            metric_results_dict_w_params.update(metric_results_dict)
            metric_results_dict_w_params["encoder_latency_s"] = encoder_latency
            metric_results_dict_w_params["decoder_latency_s"] = decoder_latency
            metric_results_dict_w_params["encoder_throughput"] = encoder_throughput
            metric_results_dict_w_params["decoder_throughput"] = decoder_throughput
            # metric_results_dict["latency_per_sample"] = throughput


            # metric_results_dict["mem_footprint_percentage"] = self.dataset_size
            # metric_results_dict["mem_footprint_number"] = self.dataset_size

            filename_csv = METRIC_REPORTS_DIR_PATH+ self.model_name + "_"+ self.dataset_name  + "_metrics.csv"
            filename_csv = filename_csv.replace("facebook/", "")
            filename_csv = filename_csv.replace("google/", "")
            if overwrite_csv or not pathlib.Path(filename_csv).exists():
                pd.DataFrame(metric_results_dict_w_params, index=[0]).to_csv(filename_csv, index=False)
                print("Created new csv file: " + filename_csv)
                # TODO: set overwrite to false
            else:
                    pd.DataFrame(metric_results_dict_w_params, index=[0]).to_csv(filename_csv, index=False, mode='a', header=False)
                    print("Appended to csv file: " + filename_csv)


        # # Get the resource utilization from the profiler 
        # res_util_df = pd.DataFrame({e.key:e.__dict__ for e in profInference.key_averages()}).T

        # columns_needed= ['count', 'cpu_time_total', 'cuda_time_total', "self_cpu_time_total","self_cuda_time_total","input_shapes","cpu_memory_usage","cuda_memory_usage","self_cpu_memory_usage","self_cuda_memory_usage","device_type","flops"]
        # res_util_df.sort_values(['cuda_time_total', 'cpu_time_total'], ascending=False)
        # if save_res_util:
        #     res_util_df.to_csv(self.model_name + "_"+ self.dataset_name + "_res_util_full.csv")
            
        #     res_util_df[columns_needed].to_csv(self.model_name + "_"+ self.dataset_name + "_res_util_summary.csv")
        #     res_util_df.head(3).to_csv(self.model_name + "_"+ self.dataset_name + "_res_util_3_rows.csv")
        
        # print(res_util_df.head(3))
    

        

    def evaluate_speed(self, batch_size=128, save_res_util=False):
        from transformers import DataCollatorForSeq2Seq
        tokenized_datasets = self.evaluation_dataset.map(
            self.preprocess_function_with_text_target,
            batched=True,
            remove_columns=self.evaluation_dataset.column_names,

        )
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        eval_dataloader = DataLoader(
                tokenized_datasets, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
        )

        

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    record_shapes=False,
                    profile_memory=True,
                    with_flops=False,
                    schedule=torch.profiler.schedule(
                            wait=2,
                            warmup=2,
                            active=3,
                            repeat=2,
                        ),
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/' + self.model_name +"_"+ self.dataset_name),

        ) as profInference:
            # TODO: torch.no_grad() or without it?
            startTime = torch.cuda.Event(enable_timing=True)
            endTime = torch.cuda.Event(enable_timing=True)
            with record_function("all_batch_inference"):
                startTime.record()
                for batch in tqdm(eval_dataloader):
                    with record_function("model_inference"):
                        with torch.no_grad():
                            # TODO: Record the time for each batch
                            # batch = {k: v.to(self.device) for k, v in batch.items()}
                            if  "nllb" in model_name:
                                # Max_length and num_beams are set to the default values as in the paper.
                                outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=128, num_beams=1)

                            else:
                                outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=128, num_beams=1)
                    profInference.step() 

                endTime.record()
                # TODO: See if the synchronization is needed
                torch.cuda.synchronize()
                latency_ms = startTime.elapsed_time(endTime) 
                latency_s = latency_ms / 1000.0
                print(f"Latency: {latency_s} seconds")
                throughput =  (len(tokenized_datasets) * batch_size) / latency_s
                print(f"Throughput: {throughput} ")

                import gc
                gc.collect()


                


    def add_prefix(self, example):
        return prefix + example[source_lang]


    def preprocess_function_with_text_target(self,examples):
       
                    
        inputs = [prefix + ex["en"] for ex in examples["translation"]]
        targets = [ex["fr"] for ex in examples["translation"]]


        padding_param = self.hyperparameters.get("tokenizer_padding_setting") 
        # print("PRINT PADDING PARAM")
        # print(padding_param)
        model_inputs = None


        if padding_param is not None and  padding_param == "do_not_pad":
            # Will not pad. This is the default behaviour
            model_inputs = self.tokenizer(
                inputs, text_target=targets,  truncation=True, padding="do_not_pad"
            )
            # print("DO NOT PAD")
        else:
            # Will pad to what is set in max_length
            model_inputs = self.tokenizer(
                inputs, text_target=targets, max_length=512, truncation=True
            )
            # print("PAD")

        return model_inputs

    
  
    def load_relevant_metrics(self, metrics_args):
        metrics_modules = {}
        for metric_name in metrics_args:
            try:
                
                metric = evaluate.load(metrics_name_map_dict[metric_name])
                metrics_modules[metric_name] = metric
            except:
                print(metric_name)
                raise NotImplementedError("Metric not supported, or may be written incorrectly.")
            
        return metrics_modules        
    

    # TODO: pass kwargs instead. make it so that any metric can be computed
    def report_metrics(self, pred_lengths, print_terminal=True):
        print("Reporting Quality Metrics: ")
        metric_results_dict = {}
        for metric_name, metric_module in self.all_metrics.items():
            if metric_name == "sacrebleu":
                result = metric_module.compute() # default: smooth_method = "exp"
                score = result["score"]
            elif metric_name == "spBleu":
                result = metric_module.compute(tokenize="flores101")
                score =  result["score"]
            elif metric_name == "chrf":
                result = metric_module.compute()
                score =  result["score"]
            elif metric_name == "chrf++" or metric_name == "chrfpp":
                result = metric_module.compute(word_order=2)
                score =  result["score"]
            elif metric_name == "meteor":
                result = metric_module.compute()
                score = result["meteor"]
            else:
                print(metric_name)
                raise NotImplementedError("Metric not supported, or may be written incorrectly")

            if print_terminal:
                print(f"Full {metric_name} metric report for this batch: {result}")

            metric_results_dict[metric_name] =score


        gen_len = np.mean(pred_lengths)
        if print_terminal:
            print(f"gen_len: {gen_len}")

            print("**************")

        metric_results_dict["gen_len"] = gen_len
        return metric_results_dict


def load_wmt14():
    pass

def load_flores200(dataset_size):
    pass


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


def retrieve_previous_experiments(configurations_sweep):
        previous_experiments = {}
        for model_name in configurations_sweep["model_name"]:
            for dataset_name in configurations_sweep["dataset_name"]:

                filename_csv = METRIC_REPORTS_DIR_PATH+ model_name + "_"+ dataset_name  + "_metrics.csv"
                filename_csv = filename_csv.replace("facebook/", "")
                filename_csv = filename_csv.replace("google/", "")
                if pathlib.Path(filename_csv).exists():
                    previous_experiments[model_name + dataset_name] = pd.read_csv(filename_csv)
        return previous_experiments



def skip_previous_experiments(previous_experiments,current_params):
    # If the experiment has already been run, skip it.
    if current_params["model_name"] + current_params["dataset_name"] in previous_experiments.keys():
        # params_df = pd.DataFrame(current_params, index=[0])
        param_cols = current_params.keys()        
        subset_params_df =previous_experiments[current_params["model_name"] + current_params["dataset_name"]][param_cols]

        # if params_df.isin(subset_params_df).all().all():
        #     print("Experiment already run. Skipping...")
        #     return True
        

        if len(subset_params_df.loc[(subset_params_df["dataset_size"] == int(current_params["dataset_size"])) & (subset_params_df["max_gen_length"] == int(current_params["max_gen_length"])) & (subset_params_df["beam_size"] == int(current_params["beam_size"])) & (subset_params_df["max_input_seq_length"] == int(current_params["max_input_seq_length"])) & (subset_params_df["tokenizer_padding_setting"] == current_params["tokenizer_padding_setting"])]) > 0:


            print("Experiment already run. Skipping...")
            return True
    return False

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    # Extensive Experimentation with different hyperparameters

    # ---- Additional Hyperparameters ----
    original_batch_size = 16
    metrics_args = ["sacrebleu", "spBleu", "chrf", "chrfpp", "meteor"]


    sweep = True
    dataset_name = "wmt14"

    continue_with_errors  = True
    counter_error = 20
    streaming = False
    save_res_util = False
    rerun_experiments = False
    # Parse the arguments
    args = utils.parse_args()

    # configurations_sweep = quality_experiment_configurations_sweep
    configurations_sweep = large_model_quality_experiments


    if sweep:
        # Retrieve previous experiments. If they exist, we will not run them again.
        previous_experiments = retrieve_previous_experiments(configurations_sweep)
        
        for model_name in configurations_sweep["model_name"]:

            print("Model Name: " + str(model_name))
            for dataset_name in configurations_sweep["dataset_name"]:
                print("--------------------------------------------------------------------------------------------------")
                print("Dataset Name: " + str(dataset_name))

                for dataset_size in configurations_sweep["dataset_size"]:
                    print("*****************")
                    
                    valid_dataset = retrieve_relevant_validation_dataset(dataset_name, dataset_size)
                    
                    
                    print("Validation Dataset Size: " + str(len(valid_dataset)))

                    evalEngine = Evaluation(model_name=model_name, 
                        evaluation_dataset=valid_dataset,
                        dataset_name=dataset_name, 
                        metrics_args = metrics_args,
                        streaming=streaming)      
                    for max_input_seq_length in configurations_sweep["max_input_seq_length"]:
                        print("Max Input Length: " + str(max_input_seq_length))

                        for tokenizer_padding_setting in configurations_sweep["tokenizer_padding_setting"]:

                            print("Tokenizer Padding Setting: " + str(tokenizer_padding_setting))


                            other_hyperparameters = {"max_input_seq_length": max_input_seq_length,
                                            "tokenizer_padding_setting": tokenizer_padding_setting,
                                            }
                            

                            evalEngine.set_other_hyperparameters(other_hyperparameters)

                            import gc
                            gc.collect()

                            torch.cuda.empty_cache()              
                            
                            for max_gen_length in configurations_sweep["max_gen_length"]:
                                print("Max Gen Length: " + str(max_gen_length)) # TODO: What is max_length here exactly??
                                for beam_size in configurations_sweep["beam_size"]:
                                    # Reset batch size to the original one
                                    batch_size = original_batch_size
                                    print("Beam Size: " + str(beam_size))
                                    text = f"Hyperparameters: model_name: {model_name}, Batch_size: {batch_size}, dataset_size: {len(valid_dataset)}, max_gen_length: {max_gen_length}, beam_size: {beam_size}, max_input_seq_length: {max_input_seq_length}, tokenizer_padding_setting: {tokenizer_padding_setting}, dataset_name= {dataset_name} \n"
                                    print(text)

                                    current_params = {"model_name": model_name,
                                                    "batch_size": batch_size,
                                                    "dataset_size": len(valid_dataset),
                                                    "max_gen_length": max_gen_length,
                                                    "beam_size": beam_size,
                                                    "max_input_seq_length": max_input_seq_length,
                                                    "tokenizer_padding_setting": tokenizer_padding_setting,
                                                    "dataset_name": dataset_name,
                                                    }
                                                    
                                    # if the experiment has already been run, skip it.
                                    if skip_previous_experiments(previous_experiments,current_params):
                                        continue

                                        
                                    try:
                                        # If experiment runs successfully, break out of the loop and continue with the next experiment.
                                        evalEngine.call_batched(batch_size=batch_size,
                                                            save_res_util=save_res_util,
                                                            beam_size=beam_size,
                                                            max_gen_length=max_gen_length,
                                                            save_metrics_csv=True, overwrite_csv=False)
                                
                                    except Exception as e:
                                        fail_logs_dir = "./fail_logs/"
                                        fail_logs_file = (fail_logs_dir + model_name + ".txt").replace("facebook/", "").replace("google/", "")
                                        failed_experiments_csv = (fail_logs_dir + "failed_experiments.csv").replace("facebook/", "").replace("google/", "")
                                        
                                        with open(failed_experiments_csv, "a") as f:
                                            csv_writer = csv.writer(f)
                                            csv_writer.writerow([model_name, batch_size, dataset_size, max_gen_length, beam_size, max_input_seq_length, tokenizer_padding_setting, dataset_name])

                                        with open(fail_logs_file, "a") as log:
                                            log.write("ERROR: Exception occured during runtime!\n")
                                            print("ERROR: Exception occured during runtime!")
                                            print(e)

                                            text = f"Hyperparameters: model_name: {model_name}, Batch_size: {batch_size}, dataset_size: {dataset_size}, max_gen_length: {max_gen_length}, beam_size: {beam_size}, max_input_seq_length: {max_input_seq_length}, tokenizer_padding_setting: {tokenizer_padding_setting}, dataset_name= {dataset_name} \n"
                                            log.write(text)
                                            log.write("Full traceback message:\n")
                                            log.write("**********")
                                            traceback.print_exc(file=log)
                                            log.write(str(torch.cuda.memory_stats(device=None)) + "\n")
                                            log.write("**********\n")
                                            log.write("End of traceback message\n")



                                            # IF it is an out of memory issue, we empty cache and retry with a smaller batch size.
                                            if "out of memory" in str(e).lower() and batch_size !=1:  
                                                log.write("Attempting to reduce batch size and retrying...\n")
                                                print("Attempting to reduce batch size and retrying...")
                                                gc.collect()
                                                torch.cuda.empty_cache()
                                                if batch_size > 2:
                                                    batch_size = batch_size/2
                                                batch_size = batch_size/2
                                                batch_size = int(batch_size)

                                                

                                            # If not out of memory issue, break out of the loop and continue with the next experiment.
                                            # Counter error added in order to give the user the option to continue with errors.
                                            elif continue_with_errors and counter_error >0:
                                                log.write("Continuing with errors...\n")
                                                print("Continuing with errors...")
                                                counter_error -=1
                                            
                                            else:
                                                log.write("Benchmark_suite failed. Please check the logs for more information.")
                                                print("Benchmark_suite failed. Please check the logs for more information.")
                                                exit(-1)
                                        


    
    if sweep:
        exit()

    
    #  ------------------------------------------



    #  ----- Testing ----- 

    # Define your model




    # streaming = False
    # save_res_util = False
    # # model_name = "google/t5-v1_1-small" # Test Model
    # model_name = "t5-base" # Test Model
    model_name = "t5-small"
    model_name = "google/switch-base-128"
    # model_name = "facebook/nllb-200-distilled-600M"
    # model_name = "t5-11b"
    # model_name = "facebook/nllb-200-1.3B"
    # model_name = "facebook/nllb-moe-54b"

    # # dataset_name = "opus100"  # Test Dataset
    dataset_name = "wmt14"
        # dataset_name = "wmt14"
    # dataset_name="facebook/flores"

    # -------------------------

    print("Model name: " + model_name)
    # # Define your evaluation dataset (FLORES)
    # # TODO: Change to FLORES
    # # TODO: Make sure to not use the same dataset that was used to train the models!!
    # all_dataset = load_dataset(dataset_name, "fr-en", split="validation", streaming=streaming)
    all_dataset = retrieve_relevant_validation_dataset(dataset_name, dataset_size=1)

    # all_dataset = load_dataset(dataset_name, "fr-en", split="test", streaming=streaming)
    print(all_dataset)
    # all_dataset = load_dataset(dataset_name, "fra_Latn", split="dev[:100]", streaming=streaming)    
    # print(all_dataset[1])
    # all_dataset = load_dataset(dataset_name, "eng_Latn", split="dev[:100]", streaming=streaming)
    # print(all_dataset[1])

    # exit()

    hyperparameters = {"max_input_seq_length": 10, "tokenizer_padding_setting": "do_not_pad"}

    evalEngine = Evaluation(model_name=model_name, evaluation_dataset=all_dataset, dataset_name=dataset_name , metrics_args = metrics_args, streaming=streaming, hyperparameters=hyperparameters)

    # evalEngine.call_batched(batch_size=32, beam_size=64,max_gen_length=512, save_res_util=save_res_util, save_metrics_csv=True, overwrite_csv=False)
    evalEngine.call_batched(batch_size=32, beam_size=64,max_gen_length=64, save_res_util=save_res_util, save_metrics_csv=True, overwrite_csv=False)
