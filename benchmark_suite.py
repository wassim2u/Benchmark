from typing import Any
import torch
from transformers import NllbTokenizer, T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration, M2M100ForConditionalGeneration, NllbMoeForConditionalGeneration
from torch.profiler import profile, record_function, ProfilerActivity

from datasets import load_dataset
from torch.utils.data import IterableDataset, Subset, DataLoader
import pandas as pd
import numpy as np
import evaluate

from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile

from tqdm.auto import tqdm
import time
import argparse
import traceback     
                
                
import pathlib

METRIC_REPORTS_DIR_PATH = "./metric_logs/"

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

max_source_length = 1024


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
    def __init__(self,model_name, evaluation_dataset, dataset_name, metrics_args, streaming=False, src_lang="en", tgt_lang="fr"):
        self.evaluation_dataset = evaluation_dataset
        self.dataset_name = dataset_name
        self.streaming = streaming
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        if src_lang == "en" and tgt_lang == "fr" and "t5" in model_name:
            self.prefix = "translate English to French: "
        else:
            self.prefix = ""
        # Define the metrics
        self.all_metrics = self.load_relevant_metrics(metrics_args)


        
        # Define model and tokenizer
        self.model_name = model_name
        if model_name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/t5-v1_1-small"]: 
            # self.tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=max_length)
            # self.tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=512)
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=512)

            self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        elif model_name in ["facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-1.3B"] and src_lang == "en" and tgt_lang == "fr":
            self.tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="fra_Latn")
            self.model= M2M100ForConditionalGeneration.from_pretrained(model_name)


        elif "nllb-moe-54b" in model_name and src_lang == "en" and tgt_lang == "fr":
            self.tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="fra_Latn")
            self.model= NllbMoeForConditionalGeneration.from_pretrained(model_name)
        else:
            raise NotImplementedError("Model not supported")


        self.model.eval()
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: Check if this is correct
        self.model.to(self.device)
        

 

        
    def __call__(self, deepspeed=False, save_profile=False, export_trace_file=False, export_flame_graph=False, *args: Any, **kwds: Any) -> Any:
        if deepspeed:
            with get_accelerator().device(0):
                flops, macs, params = get_model_profile(
                    self.model,
                    kwargs=self.preprocess_function_constructor(),
                    print_profile=True,
                    detailed=True,
                )
                print(flops)
                print(macs)
                print(params)
                exit()

                


        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as profInference:
            # TODO: torch.no_grad() or without it?
            with record_function("model_inference"):
                # TODO: Use datsets.map() instead of this function
                self.inputs = self.preprocess_function().input_ids.to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(self.inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
                decoded_labels = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        
        result = self.calculate_bleu(decoded_labels, self.evaluation_dataset["translation"])
        print(result)
        
        
        # Get the resource utilization from the profiler 

        res_util_df = pd.DataFrame({e.key:e.__dict__ for e in profInference.key_averages()}).T
        res_util_df[['count', 'cpu_time_total', 'cuda_time_total']].sort_values(['cuda_time_total', 'cpu_time_total'], ascending=False)
        if save_profile:
            res_util_df.to_csv(METRIC_REPORTS_DIR_PATH + self.model_name + "_"+ self.dataset_name + "_res_util_full.csv")
        print(res_util_df.head(10))
    
        # # Calculate Throughput:  Get the total time and number of examples from the profiler
        # print(profInference.self_cpu_time_total)
        # total_time = float(profInference.self_cpu_time_total) / 1000.0  # Convert to seconds
        # print(total_time)
        # total_examples = len(self.evaluation_dataset["translation"])


    # def call_with_nvtx(self):
    #     with torch.autograd.profiler.emit_nvtx():
    #         self.inputs = self.preprocess_function().to(self.device)
    #         with torch.no_grad():
    #             outputs = self.model.generate(self.inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    #         decoded_labels = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        


    def call_batched(self, beam_size=1, max_length=128, batch_size=128, save_res_util=False, save_metrics_csv=False, overwrite_csv=False):
        from transformers import DataCollatorForSeq2Seq


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

        scores_sacrebleu = []
        scores_chrfpp = []


        all_preds, all_labels = None, None
        pred_lengths = np.array([])

        start_time_total = time.time()
        batches_latencies = 0
    
        for batch in tqdm(eval_dataloader):
                start_time_batch = time.time()
                with torch.no_grad():
                    # batch = {k: v.to(self.device) for k, v in batch.items()}
                    if  "nllb" in model_name:
                        outputs = self.model.generate(batch["input_ids"].to(self.device), forced_bos_token_id=self.tokenizer.lang_code_to_id["fra_Latn"], do_sample=False, max_length=max_length, num_beams=beam_size)

                    else:
                        # outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=128, num_beams=1) # Works, for transformers.
                        
                        # outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=512, num_beams=4)
                        outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=max_length, num_beams=beam_size)

                        # outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=128, num_beams=4, early_stopping=True)
                        # outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=512, num_beams=4, early_stopping=True)
                        # outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=False, max_length=128, num_beams=1, early_stopping=True)

                end_time_batch = time.time()
                latency_per_batch = end_time_batch - start_time_batch
                batches_latencies += latency_per_batch
                labels = batch["labels"]
                decoded_preds, decoded_labels, pred_length  = postprocess_batch(outputs, labels, self.tokenizer)
                # all_preds = np.concatenate((all_preds, decoded_preds), axis=0) if all_preds is not None else decoded_preds
                # all_labels = np.concatenate((all_labels, decoded_labels), axis=0) if all_labels is not None else decoded_labels
                # print(decoded_preds)
                # print(decoded_labels)
                # print("--")
                for metric_name in self.all_metrics.keys():
                    self.all_metrics[metric_name].add_batch(predictions=decoded_preds, references=decoded_labels)

                pred_lengths = np.concatenate((pred_lengths, pred_length), axis=0) if len(pred_lengths) !=0 else pred_length
                
                # result_bleu = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels, tokenize="intl", smooth_method="exp")

                # result_chrfpp = chrf.compute( word_order=2, predictions=decoded_preds, references=decoded_labels, smooth_method="exp")
                # print(f"Full BLEU score report for this batch: {result_bleu}")
                # scores_sacrebleu.append(result_bleu["score"])
                # scores_chrfpp.append(result_chrfpp.score)
                decoded_preds, decoded_labels = None, None
                

        end_time_total = time.time()
        latency = end_time_total - start_time_total
        print(str(latency - batches_latencies) + "diff overhead in seconds")
        print(str(batches_latencies) + "batch_latencies seconds")
        print(str(latency) + "latency total runtime seconds")

        # Export to 
        # profInference.export_chrome_trace("trace.json")

        # print(all_preds[0])

        # result_bleu = sacrebleu.compute(smooth_method="exp")
        
        metric_results_dict = self.report_metrics(pred_lengths)
        end_time_total_2 = time.time()
        latency_2 = end_time_total_2 - start_time_total
        print(str(latency_2 - latency) + "diff overhead with metric reporting in seconds")
        print(str(latency_2) + "total runtime with decoding time in seconds")

        if save_metrics_csv:
            metric_results_dict_w_params = {}
            metric_results_dict_w_params["model_name"] = self.model_name
            total_params = sum(p.numel() for p in self.model.parameters())
            metric_results_dict_w_params["total_params"] = total_params
            metric_results_dict_w_params["dataset_name"] = self.dataset_name
            metric_results_dict_w_params["dataset_size"] = self.valid_split
            metric_results_dict_w_params["beam_size"] = beam_size
            metric_results_dict_w_params["batch_size"] = batch_size
            metric_results_dict_w_params["max_length"] = max_length
            metric_results_dict_w_params["max_source_length"] = max_source_length
            metric_results_dict_w_params["src_lang"] = self.src_lang
            metric_results_dict_w_params["tgt_lang"] = self.tgt_lang
            metric_results_dict_w_params.update(metric_results_dict)
            # metric_results_dict["dataset_size"] = self.dataset_size
            # metric_results_dict["latency_s"] = latency
            # metric_results_dict["througput"] = throughput
            # metric_results_dict["latency_per_sample"] = throughput


            # metric_results_dict["mem_footprint_percentage"] = self.dataset_size
            # metric_results_dict["mem_footprint_number"] = self.dataset_size
            # metric_results_dict["dataset_size"] = self.dataset_size
            # metric_results_dict["dataset_size"] = self.dataset_size

            if overwrite_csv or not pathlib.Path(METRIC_REPORTS_DIR_PATH + self.model_name + "_"+ self.dataset_name +  "_metrics.csv").exists():
                pd.DataFrame(metric_results_dict, index=[0]).to_csv(METRIC_REPORTS_DIR_PATH+ self.model_name + "_"+ self.dataset_name  + "_metrics.csv")
                print("Created new csv file")
            else:
                    pd.DataFrame(metric_results_dict, index=[0]).to_csv(METRIC_REPORTS_DIR_PATH + self.model_name + "_"+ self.dataset_name + "_metrics.csv", mode='a', header=False)
                    print("Appended to csv file")
        # result_bleu = sacrebleu.compute(smooth_method="exp")
        # result_spbleu = spBleu.compute(smooth_method="exp", tokenize = "flores200")
        # result_chrf = sacrebleu.compute()
        # result_chrfpp = sacrebleu.compute(word_order=2)


        # result_chrfpp = chrf.compute( word_order=2, predictions=decoded_preds, references=decoded_labels, smooth_method="exp")



        # avg_score_bleu = sum(scores_sacrebleu) / len(scores_sacrebleu)
        # avg_score_chrfpp = sum(scores_chrfpp) / len(scores_chrfpp)

        # print(f"BLEU score: {avg_score_bleu:.3f}")
        # print(f"chrf++ score: {avg_score_chrfpp:.2f}")


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
                tokenized_datasets, collate_fn=data_collator, batch_size=batch_size
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
                                outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=True,seed=42, max_length=512, num_beams=4)

                            else:
                                outputs = self.model.generate(batch["input_ids"].to(self.device), do_sample=True, seed=42, max_length=512, early_stopping=True, num_beams=1)
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
        model_inputs = self.tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        return model_inputs

    
  
    def load_relevant_metrics(self, metrics_args):
        metrics_modules = {}
        for metric_name in metrics_args:
            try:
                
                metric = evaluate.load(metrics_name_map_dict[metric_name])
                metrics_modules[metric_name] = metric
            except:

                raise NotImplementedError("Metric not supported, or may be written incorrectly.")
            
        return metrics_modules        
    

    # TODO: pass kwargs instead. make it so that any metric can be computed
    def report_metrics(self, pred_lengths):
        print("**************")
        print("Reporting Quality Metrics: ")
        metric_results_dict = {}
        for metric_name, metric_module in self.all_metrics.items():
            if metric_name == "sacrebleu":
                result = metric_module.compute() # default: smooth_method = "exp"
            elif metric_name == "spBleu":
                result = metric_module.compute(tokenize="flores200")
            elif metric_name == "chrf":
                result = metric_module.compute()
            elif metric_name == "chrf++" or metric_name == "chrfpp":
                result = metric_module.compute(word_order=2)
            elif metric_name == "meteor":
                result = metric_module.compute()
            else:
                raise NotImplementedError("Metric not supported, or may be written incorrectly")

            print(f"Full {metric_name} score report for this batch: {result}")

            metric_results_dict[metric_name] = result["score"]


        gen_len = np.mean(pred_lengths)
        print(f"gen_len: {gen_len}")

        print("**************")

        metric_results_dict["gen_len"] = gen_len
        return metric_results_dict

experiment_configurations_sweep = {
    "dataset_size" : ["1", "all"],
    "max_length" : [32,64,128,256,512], # Maximum generation sequence length
    "max_input_length": [15,], # Maximum input sequence length: Short sentences <= 15, Medium  15 < x< 25 , Long => 25 
    "beam_size" : [1,2,4],
    "model_name" : ["t5-small", "t5-base", "t5-large", "t5-11b" , "nllb-200-distilled-600M", "nllb-200-1.3B"],
    "dataset_name" : ["wmt14"],
    # "batch_size" : ,
}

experiment_configurations_sweep_small = {
    "dataset_size" : ["1","100", "1000", "all"],
    "max_length" : [32,64,128,256,512], # Maximum sequence length
    "beam_size" : [1,2,4],
    "model_name" : ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b" , "nllb-200-distilled-600M"],
    "dataset_name" : ["wmt14"],
    # "batch_size" : ,
}




experiment_configurations_sweep_test = {
    "dataset_size" : ["1","100"],
    "max_length" : [16,512], # Maximum  sequence length
    "model_name" : ["t5-small", "t5-base"],
}

# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--predict_with_generate",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )


    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``."
        ),
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

   
    

    parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the evaluation dataloader.",
    )
  
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()


    if args.dataset_name is None:
        raise ValueError("Please include a dataset name to run evaluation on")

    return args


if __name__ == "__main__":
    # ---- Additional Hyperparameters ----
    batch_size = 64
    metrics_args = ["sacrebleu", "spBleu", "chrf", "chrfpp", "meteor"]


    sweep = False
    dataset_name = "wmt14"

    streaming = False
    save_res_util = False
    rerun_experiments = False
    # Parse the arguments
    args = parse_args()


    if sweep:

        try:
            for dataset_size in experiment_configurations_sweep["dataset_size"]:
                print("*****************")
                valid_split = None
                if dataset_size == "all":
                    valid_split = "validation"
                else:
                    valid_split = "validation[:{}]".format(dataset_size)
                
                print("Dataset Size: " + str(dataset_size))
                valid_dataset = load_dataset(dataset_name, "fr-en", split="validation", streaming=False)
                for model_name in experiment_configurations_sweep["model_name"]:
                    print("Model Name: " + str(model_name))
                    evalEngine = Evaluation(model_name=model_name, evaluation_dataset=valid_dataset, dataset_name=dataset_name , metrics_args = metrics_args, streaming=streaming)                    
                    
                    for max_length in experiment_configurations_sweep["max_length"]:
                        print("Max Length: " + str(max_length)) # TODO: What is max_length here exactly??
                        for beam_size in experiment_configurations_sweep["beam_size"]:
                            print("Beam Size: " + str(beam_size))
                            
                            evalEngine.call_batched(batch_size=32, save_res_util=save_res_util, beam_size=beam_size, max_length=max_length, save_metrics_csv=True, overwrite_csv=False)
                            # evalEngine.evaluate_speed(save_res_util=save_res_util, batch_size =16)
                    
        except RuntimeError as e:
            
            with open("./fail_logs/{}.txt", "w") as log:
                log.write("ERROR: Exception occured during runtime!")
                text = f"Hyperparameters: model_name: {model_name}, Batch_size: {batch_size}, dataset_size: {valid_split}, max_length: {max_length}, beam_size: {beam_size}"
                log.write(text)
                log.write("Full traceback message:")
                log.write("**********")
                traceback.print_exc(file=log)
                log.write("**********")
                log.write("End of traceback message")

                if "out of memory" in str(e.lower()) and batch_size !=1:  
                    log.write("Attempting to reduce batch size and retrying...")
                    batch_size = batch_size/2
                else:
                    print("Batch size cannot be reduced further")
                    exit(-1)
            
        exit()
            

    


    # Define your model




    # streaming = False
    # save_res_util = False
    # # model_name = "google/t5-v1_1-small" # Test Model
    # model_name = "t5-base" # Test Model
    model_name = "t5-small"
    # # model_name = switch
    # model_name = "facebook/nllb-200-distilled-600M"
    # model_name = "t5-11b"
    model_name = "facebook/nllb-200-1.3B"
    # model_name = "facebook/nllb-moe-54b"

    # # dataset_name = "opus100"  # Test Dataset
    dataset_name = "wmt14"
    # dataset_name="facebook/flores"

    # -------------------------


    # # Define your evaluation dataset (FLORES)
    # # TODO: Change to FLORES
    # # TODO: Make sure to not use the same dataset that was used to train the models!!
    all_dataset = load_dataset(dataset_name, "fr-en", split="validation", streaming=streaming)
    # all_dataset = load_dataset(dataset_name, "fr-en", split="test", streaming=streaming)
    print(all_dataset[1])
    # all_dataset = load_dataset(dataset_name, "fra_Latn", split="dev[:100]", streaming=streaming)    
    # print(all_dataset[1])
    # all_dataset = load_dataset(dataset_name, "eng_Latn", split="dev[:100]", streaming=streaming)
    # print(all_dataset[1])

    # exit()


    evalEngine = Evaluation(model_name=model_name, evaluation_dataset=all_dataset, dataset_name=dataset_name , metrics_args = metrics_args, streaming=streaming)

    evalEngine.call_batched(batch_size=4, save_res_util=save_res_util, save_metrics_csv=False, overwrite_csv=False)

    # # evalEngine(deepspeed=False)

    # # evalEngine.call_with_nvtx()


# TESTING 

    # text = all_dataset["translation"][0]['en']
    
    # # text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
    # text = "translate English to French: " + text
    # print(text)
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name)   
    # model.eval()
    # input_ids = tokenizer(text, return_tensors="pt").input_ids

    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # print(all_dataset["translation"][0]['fr'])


# TESTING (MoE)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # text = all_dataset["translation"][0]['en']
    
    # # text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
    # print(text)
    # tokenizer = NllbTokenizer.from_pretrained(model_name)
    # model = NllbMoeForConditionalGeneration.from_pretrained(model_name)   
    # model.eval()
    # model.to(device)

    # input_ids = tokenizer(text, return_tensors="pt").input_ids
    # input_ids = input_ids.to(device)
    # # TODO: Check if this is correct
    # print("Generating")

    # outputs = model.generate(input_ids)
    # # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # res = tokenizer.batch_decode(outputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=50, skip_special_tokens=True)
    # print(res)
    # print(res[0])
    #     # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # print(all_dataset["translation"][0]['fr'])


