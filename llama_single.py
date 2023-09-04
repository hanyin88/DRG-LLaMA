# Adopted framework from: https://github.com/tloen/alpaca-lora


import os
import pandas as pd
from datetime import datetime
import json
from typing import List

import fire
import torch
from datasets import load_dataset

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import LlamaTokenizer, LlamaForSequenceClassification
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    TaskType
)
import torch

from utils.eval_utils import cls_metrics
from utils.gen_utils import create_folder


with open('paths.json', 'r') as f:
        path = json.load(f)
        train_set_path = path["train_set_path"]
        test_set_path = path["test_set_path"]
        catche_path = path["catche_path"]
        output_path = path["output_path"]

def train(
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    model_size: str = "7b",
    train_data_path: str = train_set_path,
    val_data_path: str = test_set_path,
    split: int = 100,
    cache_dir: str = catche_path,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,# 3e-4 is the learning rate used in the LLaMA paper
    cutoff_len: int = 512, 
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "score"
    ],
    padding_side: str = "right", 
    wandb_project: str = "classification", 
    wandb_watch: str = "gradients",  
    wandb_log_model: str = "",  
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    
    now = datetime.now()
    date_string = now.strftime("%B-%d-%H-%M")
    wandb_run_name = f"{model_size}-{cutoff_len}-{micro_batch_size}-{learning_rate}-{padding_side}-{date_string}"
    output_dir = create_folder(f'{output_path}/{wandb_project}', wandb_run_name)

    # load file from train_data_path and find out the unique number of labels
    num_labels = pd.read_csv(train_data_path).label.nunique()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LLaMA-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"model_size: {model_size}\n"
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
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"padding_side: {padding_side}\n"
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


    model = LlamaForSequenceClassification.from_pretrained(
        base_model, 
        num_labels=num_labels, 
        load_in_8bit=True, 
        torch_dtype=torch.float16,  
        device_map=device_map,
        cache_dir=cache_dir)

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model, 
        model_max_length=cutoff_len,
        cache_dir=cache_dir)

    # This is to fix the bad token in "decapoda-research/llama-7b-hf"

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    model = prepare_model_for_int8_training(model)

    # note when passing task type as string argument, it will lead to error. May consider adding module_to_save manually
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=None,
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    train_data = load_dataset("csv", data_files=train_data_path, split=f'train[:{split}%]')
    test_data = load_dataset("csv", data_files=val_data_path, split=f'train[:{split}%]')

    # train_data= train_data.shard(num_shards=20000, index=0)
    # test_data= test_data.shard(num_shards=500, index=0)

    tokenized_train = train_data.map(preprocess_function, batched=True).remove_columns(["text"]).rename_column("label", "labels")
    tokenized_test = test_data.map(preprocess_function, batched=True).remove_columns(["text"]).rename_column("label", "labels")

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
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
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