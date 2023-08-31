import argparse 
from pynvml import *
import getpass
import csv
import torch
import transformers
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
import time
import functools
from typing import Any, Callable


CONFIG = {"nvme_path": f"/mnt/{getpass.getuser()}/test-data"}
METRIC_REPORTS_DIR_PATH = "./metric_logs/"



# TODO: Provide a more uniform and robust way to map files and format the directory.#
#Used for latency measurements for forward_decorator
FORWARD_TIMES_FILENAMES = {
    "t5-small" : METRIC_REPORTS_DIR_PATH + "t5_latencies.csv",
    "t5-base" : METRIC_REPORTS_DIR_PATH + "t5_latencies.csv",
    "t5-large": METRIC_REPORTS_DIR_PATH + "t5_latencies.csv",
    "nllb-200-1.3B" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "nllb-200-distilled-600M" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "facebook/nllb-moe-54b" : METRIC_REPORTS_DIR_PATH + "nllb-moe_latencies.csv",
    "google/switch-base-128" : METRIC_REPORTS_DIR_PATH + "switch_latencies.csv",
    "facebook/nllb-200-distilled-1.3B" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "facebook/nllb-200-distilled-600M" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "facebook/nllb-200-1.3B" : METRIC_REPORTS_DIR_PATH + "nllb_latencies.csv",
    "./opus_switch_model_8/checkpoint-1500" : METRIC_REPORTS_DIR_PATH + "switch_latencies.csv",
    "./opus_switch_model_16/checkpoint-6000" : METRIC_REPORTS_DIR_PATH + "switch_latencies.csv",
    "google/switch_16_finetuned" : METRIC_REPORTS_DIR_PATH + "switch_latencies.csv",

}


def setup_environment():
    num_gpus = torch.cuda.device_count()
    for gid in range(num_gpus):
        torch.cuda.set_device(gid)
        if torch.cuda.utilization() == 0:
            break
setup_environment()




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


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


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
        print("No dataset name passed to the arguments. We will default to running on wmt14 dataset.")
        args.dataset_name = "wmt14"
    return args