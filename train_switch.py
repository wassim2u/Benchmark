
from transformers import AutoTokenizer, TextStreamer


import evaluate
# Set random seed to 42
import torch
import random
import torch
torch.manual_seed(42)
random.seed(42)
checkpoint = "google/switch-base-16"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
streamer = TextStreamer(tokenizer)

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "


metric = evaluate.load("sacrebleu")


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True)
    return model_inputs



from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")

books = books["train"].train_test_split(test_size=0.2)
tokenized_books = books.map(preprocess_function, batched=True)

print(books["train"][1])

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)




import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
print(model.config.max_length)

training_args = Seq2SeqTrainingArguments(
    output_dir="opus_switch_model_16",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    generation_max_length=512
)
training_args = training_args.set_logging(strategy="steps", steps=100, report_to=["tensorboard"])

training_args.logging_dir = 'train_logs/' # or any dir you want to save logs


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()

trainer.save_model("switch-base-16-finetuned")



# compute train results
metrics = train_result.metrics
max_train_samples = len(tokenized_books["train"])
metrics["train_samples"] = min(max_train_samples, len(tokenized_books["train"]))

# save train results
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)






train_loss = []
for elem in training_args.state.log_history:
    if 'loss' in elem.keys():
        train_loss.append(elem['loss'])

# compute evaluation results
metrics = trainer.evaluate()
max_val_samples = len(tokenized_books["test"])
metrics["eval_samples"] = min(max_val_samples, len(tokenized_books["test"]))

# save evaluation results
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)