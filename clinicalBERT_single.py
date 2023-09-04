import os
from typing import List
import pandas as pd
from datetime import datetime


import fire
import torch
from datasets import load_dataset

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

from huggingface_hub import notebook_login
import numpy as np

from utils.eval_utils import cls_metrics
from utils.gen_utils import create_folder


with open('paths.json', 'r') as f:
        path = json.load(f)
        train_set_path = path["train_set_path"]
        test_set_path = path["test_set_path"]
        catche_path = path["catche_path"]
        output_path = path["output_path"]

def train(
    base_model: str = "emilyalsentzer/Bio_ClinicalBERT",  # the only required argument
    train_data_path: str = train_set_path,
    val_data_path: str = test_set_path,
    cache_dir: str = catche_path,
    split: int = 100,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,# 3e-4 is the learning rate used in the LLaMA paper
    cutoff_len: int = 512, # consider changing to 1024
    model_name: str = "bert",
    wandb_project: str = "classification", #other options: "generative", "multilabel-classification",
    wandb_watch: str = "gradients",  # options: false | gradients | all ; issues when using all: I have since bypassed this issue by only logging gradient and instead of all.
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    
    now = datetime.now()
    date_string = now.strftime("%B-%d-%H-%M")
    wandb_run_name = f"{model_name}-{cutoff_len}-{micro_batch_size}-{learning_rate}-{date_string}"
    output_dir = create_folder(f'{output_path}/{wandb_project}', wandb_run_name)

    # load file from train_data_path and find out the unique number of labels
    num_labels = pd.read_csv(train_data_path).label.nunique()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LLaMA-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_data_path: {train_data_path}\n"
            f"val_data_path: {val_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"cache_dir: {cache_dir}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"split: {split}\n"
            f"num_labels: {num_labels}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model


    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, 
        num_labels=num_labels, 
        cache_dir=cache_dir)


    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        model_max_length=cutoff_len,
        cache_dir=cache_dir)


    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    print_trainable_parameters(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    train_data = load_dataset("csv", data_files=train_data_path, split=f'train[:{split}%]')
    test_data = load_dataset("csv", data_files=val_data_path, split=f'train[:{split}%]')

    train_data= train_data.shard(num_shards=5000, index=0)
    test_data= test_data.shard(num_shards=2000, index=0)

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_test = test_data.map(preprocess_function, batched=True)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return cls_metrics(predictions, labels, class_num=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        push_to_hub=False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)