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
from archer.runtime import ArcherEngine
import getpass
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.nllb_moe.modeling_nllb_moe import (
    NllbMoeEncoder,
    NllbMoeDecoder,
)
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersStack, )
import transformers


# from deepspeed.accelerator import get_accelerator
# from deepspeed.profiling.flops_profiler import get_model_profile
import utils
from hyperparameters_config import quality_experiment_configurations_sweep

from tqdm.auto import tqdm
import time
import argparse
import traceback     
                
                
import pathlib


CONFIG = {"nvme_path": f"/mnt/{getpass.getuser()}/test-data"}
METRIC_REPORTS_DIR_PATH = "./metric_logs/"


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
                                                 NllbMoeEncoder) else "decoder"
            
        
        with open(path_to_logfile, 'a') as f:
            # f.write(f"{layer_type},{end_time - start_time}\n")

            writer = csv.writer(f)
            writer.writerow([layer_type, end_time - start_time])


        # print(f"{layer_type} forward time: {end_time - start_time}s")
        return result

    return timer_forward





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


max_length = 128

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
        self.PATH_TO_LOG_LATENCY = METRIC_REPORTS_DIR_PATH + self.model_name + "_"+ self.dataset_name + "_latencies.csv"
        self.add_latency_measurement_functionality( path_to_log_latency =  self.PATH_TO_LOG_LATENCY)

        
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
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=512)
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
        self.model.to(self.device)
        



    def add_latency_measurement_functionality(self, path_to_log_latency):
        if pathlib.Path(path_to_log_latency).exists():
            import os
            os.remove(path_to_log_latency)
        transformers.models.t5.modeling_t5._old_t5_forward = T5Stack.forward
        transformers.models.nllb_moe.modeling_nllb_moe._old_nllb_moe_encoder_forward = (
            NllbMoeEncoder.forward)
        transformers.models.nllb_moe.modeling_nllb_moe._old_nllb_moe_decoder_forward = (
            NllbMoeDecoder.forward)
        transformers.models.switch_transformers._old_switch_transformers_forward = (
            SwitchTransformersStack.forward)
        




        T5Stack.forward = forward_decorator(T5Stack.forward, path_to_log_latency)
        NllbMoeEncoder.forward = forward_decorator(NllbMoeEncoder.forward, path_to_log_latency)
        NllbMoeDecoder.forward = forward_decorator(NllbMoeDecoder.forward, path_to_log_latency)
        SwitchTransformersStack.forward = forward_decorator(
            SwitchTransformersStack.forward, path_to_log_latency)

        

        
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

    def call_batched(self, beam_size=1, 
                    max_gen_length=128,
                    max_input_length=128,
                    tokenizer_padding_setting="pad_to_max_length", 
                    batch_size=128, 
                    pytorch_profiling=False, save_res_util=False, save_metrics_csv=False, overwrite_csv=False):
        from transformers import DataCollatorForSeq2Seq
            
        print(len(self.evaluation_dataset))
        if self.hyperparameters.get("max_input_length") is not None and len(self.evaluation_dataset) != 1:

            self.evaluation_dataset = self.evaluation_dataset.filter(lambda example: (len(example["translation"]["en"])) <= self.hyperparameters["max_input_length"])
        print(len(self.evaluation_dataset))

        tokenized_datasets = self.evaluation_dataset.map(
            self.preprocess_function_with_text_target,
            batched=True,
            remove_columns=self.evaluation_dataset.column_names,
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        # TODO: should i add label_pad_token_id??
        eval_dataloader = DataLoader(
                tokenized_datasets, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
        )


        pred_lengths = np.array([])
        # Call config to retrieve pad_token_id. Will be useful for calculating throughput.
        config = AutoConfig.from_pretrained(self.model_name)
        number_of_total_encoded_tokens = 0
        number_of_total_decoded_tokens = 0

        if not pytorch_profiling:
            
            
            for batch in tqdm(eval_dataloader):
                    input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                    

                    with torch.no_grad():
                        if  "nllb" in model_name:
                            outputs = self.model.generate(input_ids, forced_bos_token_id=self.tokenizer.lang_code_to_id["fra_Latn"], do_sample=False, max_length=max_gen_length, num_beams=beam_size)
                        elif "google/switch" in model_name:
                                
                            outputs = self.model.generate(input_ids, do_sample=False, max_length=max_gen_length, num_beams=beam_size,   decoder_start_token_id=0,
                                                 bos_token_id=0)

                        else:
                            # outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=128, num_beams=1) # Works, for transformers.
                            outputs = self.model.generate(input_ids, do_sample=False, max_length=max_gen_length, num_beams=beam_size)

                    
                    
                    labels = batch["labels"]
                    decoded_preds, decoded_labels, pred_length  = postprocess_batch(outputs, labels, self.tokenizer)
            
                    for metric_name in self.all_metrics.keys():
                        self.all_metrics[metric_name].add_batch(predictions=decoded_preds, references=decoded_labels)

                    pred_lengths = np.concatenate((pred_lengths, pred_length), axis=0) if len(pred_lengths) !=0 else pred_length
                
                    decoded_preds, decoded_labels = None, None

                    # For calculating throughput, we need to know the number of total tokens generated.
                    attention_mask = (input_ids != config.pad_token_id).to(input_ids.dtype)
                    number_of_total_encoded_tokens += torch.sum(attention_mask)
                    number_of_total_decoded_tokens += np.sum(pred_length)

        else:
            pass
          
                    


        # Export to 
        # profInference.export_chrome_trace("trace.json")


        # Retrieve and process latency from log file
        latencies_df = pd.read_csv(self.PATH_TO_LOG_LATENCY, names=["layer_type", "latency_s"]).groupby("layer_type").mean() 
        print(latencies_df)
        encoder_latency = latencies_df.loc["encoder"]["latency_s"]
        decoder_latency = latencies_df.loc["decoder"]["latency_s"]
        print(f"Encoder latency: {encoder_latency}")
        print(f"Decoder latency: {decoder_latency}")

        # Calculate throughput
        encoder_throughput = (number_of_total_encoded_tokens/ encoder_latency).cpu().numpy()
        decoder_throughput = (number_of_total_decoded_tokens/ decoder_latency)
        print(f"Encoder throughput: {encoder_throughput}")
        print(f"Decoder throughput: {decoder_throughput}")

        metric_results_dict = self.report_metrics(pred_lengths)

        if save_metrics_csv:
            metric_results_dict_w_params = {}
            metric_results_dict_w_params["model_name"] = self.model_name
            total_params = sum(p.numel() for p in self.model.parameters())
            metric_results_dict_w_params["total_params"] = total_params
            metric_results_dict_w_params["dataset_name"] = self.dataset_name
            metric_results_dict_w_params["dataset_size"] = len(self.evaluation_dataset)
            metric_results_dict_w_params["src_lang"] = self.src_lang
            metric_results_dict_w_params["tgt_lang"] = self.tgt_lang
            metric_results_dict_w_params["batch_size"] = batch_size
            metric_results_dict_w_params["beam_size"] = beam_size
            metric_results_dict_w_params["max_gen_length"] = max_gen_length
            metric_results_dict_w_params["max_input_seq_length"] = max_input_length # TODO: Report on different sentence lengths.
            metric_results_dict_w_params["tokenizer_padding_setting"] = tokenizer_padding_setting # TODO: Report on different sentence lengths.

            metric_results_dict_w_params.update(metric_results_dict)
            metric_results_dict_w_params["encoder_latency_s"] = encoder_latency
            metric_results_dict_w_params["decoder_latency_s"] = decoder_latency
            metric_results_dict_w_params["encoder_throughput"] = encoder_throughput
            metric_results_dict_w_params["decoder_throughput"] = decoder_throughput
            # metric_results_dict["latency_per_sample"] = throughput


            # metric_results_dict["mem_footprint_percentage"] = self.dataset_size
            # metric_results_dict["mem_footprint_number"] = self.dataset_size


            if overwrite_csv or not pathlib.Path(METRIC_REPORTS_DIR_PATH + self.model_name + "_"+ self.dataset_name +  "_metrics.csv").exists():
                pd.DataFrame(metric_results_dict_w_params, index=[0]).to_csv(METRIC_REPORTS_DIR_PATH+ self.model_name + "_"+ self.dataset_name  + "_metrics.csv", index=False)
                print("Created new csv file")
                # TODO: set overwrite to false
            else:
                    pd.DataFrame(metric_results_dict_w_params, index=[0]).to_csv(METRIC_REPORTS_DIR_PATH + self.model_name + "_"+ self.dataset_name + "_metrics.csv", index=False, mode='a', header=False)
                    print("Appended to csv file")


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
        model_inputs = None

        if padding_param is not None and  padding_param == "do_not_pad":
            # Will not pad. This is the default behaviour
            model_inputs = self.tokenizer(
                inputs, text_target=targets,  truncation=True, padding="do_not_pad"
            )
        else:
            # Will pad to what is set in max_length
            model_inputs = self.tokenizer(
                inputs, text_target=targets, max_length=512, truncation=True
            )
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
    def report_metrics(self, pred_lengths):
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

            print(f"Full {metric_name} metric report for this batch: {result}")

            metric_results_dict[metric_name] =score


        gen_len = np.mean(pred_lengths)
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
            valid_split = "validation[:{}]".format(dataset_size)
        
        valid_dataset = load_dataset(dataset_name, "fr-en", split=valid_split)
    elif dataset_name == "flores200" or dataset_name == "facebook/flores":
        if dataset_size == "all":
            valid_split = "dev"
        else:
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

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.seed(42)
    # Extensive Experimentation with different hyperparameters

    # ---- Additional Hyperparameters ----
    batch_size = 64
    metrics_args = ["sacrebleu", "spBleu", "chrf", "chrfpp", "meteor"]


    sweep = True
    dataset_name = "wmt14"

    streaming = False
    save_res_util = False
    rerun_experiments = False
    # Parse the arguments
    args = utils.parse_args()


    if sweep:

        try:
            for dataset_name in quality_experiment_configurations_sweep["dataset_name"]:
                print("*****************")
                print("Dataset Name: " + str(dataset_name))
                for dataset_size in quality_experiment_configurations_sweep["dataset_size"]:
                    print("*****************")
                    
                    valid_dataset = retrieve_relevant_validation_dataset(dataset_name, dataset_size)
                    
                    
                    print("Validation Dataset Size: " + str(len(valid_dataset)))
                    for max_input_length in quality_experiment_configurations_sweep["max_input_length"]:
                        print("Max Input Length: " + str(max_input_length))

                        for tokenizer_padding_setting in quality_experiment_configurations_sweep["tokenizer_padding_setting"]:

                            print("Tokenizer Padding Setting: " + str(tokenizer_padding_setting))
                            for model_name in quality_experiment_configurations_sweep["model_name"]:
                                print("Model Name: " + str(model_name))

                                hyperparameters = {"max_input_length": max_input_length, "tokenizer_padding_setting": tokenizer_padding_setting}
                                
                                evalEngine = Evaluation(model_name=model_name, 
                                                        evaluation_dataset=valid_dataset,
                                                        dataset_name=dataset_name, 
                                                        metrics_args = metrics_args,
                                                        hyperparameters=hyperparameters,
                                                        streaming=streaming)                    
                                
                                for max_gen_length in quality_experiment_configurations_sweep["max_length"]:
                                    print("Max Length: " + str(max_length)) # TODO: What is max_length here exactly??
                                    for beam_size in quality_experiment_configurations_sweep["beam_size"]:
                                        print("Beam Size: " + str(beam_size))
                                    
                                        evalEngine.call_batched(batch_size=32,
                                                                save_res_util=save_res_util,
                                                                beam_size=beam_size,
                                                                max_gen_length=max_gen_length,
                                                                save_metrics_csv=True, overwrite_csv=False)

                        
        except RuntimeError as e:
            
            with open("./fail_logs/{}.txt".format(model_name), "w") as log:
                log.write("ERROR: Exception occured during runtime!")
                text = f"Hyperparameters: model_name: {model_name}, Batch_size: {batch_size}, dataset_size: {dataset_size}, max_length: {max_length}, beam_size: {beam_size}"
                log.write(text)
                log.write("Full traceback message:")
                log.write("**********")
                traceback.print_exc(file=log)
                log.write("**********")
                log.write("End of traceback message")

                if "out of memory" in str(e).lower() and batch_size !=1:  
                    log.write("Attempting to reduce batch size and retrying...")
                    batch_size = batch_size/2
                else:
                    print("Benchmark_suite failed. Please check the logs for more information.")
                    exit(-1)
            
        exit()
            

    
    #  ------------------------------------------



    #  ----- Testing ----- 

    # Define your model




    # streaming = False
    # save_res_util = False
    # # model_name = "google/t5-v1_1-small" # Test Model
    # model_name = "t5-base" # Test Model
    model_name = "t5-small"
    # model_name = "google/switch-base-128"
    # model_name = "facebook/nllb-200-distilled-600M"
    # model_name = "t5-11b"
    # model_name = "facebook/nllb-200-1.3B"
    # model_name = "facebook/nllb-moe-54b"

    # # dataset_name = "opus100"  # Test Dataset
    # dataset_name = "wmt14"
        # dataset_name = "wmt14"
    dataset_name="facebook/flores"

    # -------------------------

    print("Model name: " + model_name)
    # # Define your evaluation dataset (FLORES)
    # # TODO: Change to FLORES
    # # TODO: Make sure to not use the same dataset that was used to train the models!!
    # all_dataset = load_dataset(dataset_name, "fr-en", split="validation", streaming=streaming)
    all_dataset = retrieve_relevant_validation_dataset(dataset_name, dataset_size=10)

    # all_dataset = load_dataset(dataset_name, "fr-en", split="test", streaming=streaming)
    print(all_dataset)
    # all_dataset = load_dataset(dataset_name, "fra_Latn", split="dev[:100]", streaming=streaming)    
    # print(all_dataset[1])
    # all_dataset = load_dataset(dataset_name, "eng_Latn", split="dev[:100]", streaming=streaming)
    # print(all_dataset[1])

    # exit()


    evalEngine = Evaluation(model_name=model_name, evaluation_dataset=all_dataset, dataset_name=dataset_name , metrics_args = metrics_args, streaming=streaming)

    evalEngine.call_batched(batch_size=64, save_res_util=save_res_util, save_metrics_csv=True, overwrite_csv=False)



