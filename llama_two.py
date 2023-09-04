import os
from typing import List
import pandas as pd
from datetime import datetime


import fire
import torch
from datasets import load_dataset, Dataset

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import LlamaTokenizer, LlamaForSequenceClassification

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from utils.eval_utils import cls_metrics_multi

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    TaskType
)
import torch
import json
import numpy as np

from utils.gen_utils import create_folder

with open('paths.json', 'r') as f:
        path = json.load(f)
        multi_train_set_path = path["multi_train_set_path"]
        multi_test_set_path = path["multi_test_set_path"]
        catche_path = path["catche_path"]
        output_path = path["output_path"]
        drg_34_dissection_path = path["drg_34_dissection_path"]

def train(
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    model_size: str = "7b",
    train_data_path: str = multi_train_set_path,
    val_data_path: str = multi_test_set_path,
    drg_mapping_path: str = drg_34_dissection_path,
    cache_dir: str = catche_path,
    split: int = 100,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,# 3e-4 is the learning rate used in the LLaMA paper
    cutoff_len: int = 512, # consider changing to 1024
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
    wandb_project: str = "multilabel-classification", #other options: "generative", "multilabel-classification",
    wandb_watch: str = "gradients",  # options: false | gradients | all ; issues when using all: I have since bypassed this issue by only logging gradient and instead of all.
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    
    now = datetime.now()
    date_string = now.strftime("%B-%d-%H-%M")
    wandb_run_name = f"{model_size}-{cutoff_len}-{micro_batch_size}-{learning_rate}-{padding_side}-{date_string}"
    output_dir = create_folder(f'{output_path}/{wandb_project}', wandb_run_name)

    # load file from train_data_path and find out the unique number of labels
    num_labels_pc = pd.read_csv(drg_mapping_path).principal_diagnosis_lable.nunique()
    num_labels_cc = pd.read_csv(drg_mapping_path)["CC/MCC"].nunique()
    num_labels = num_labels_pc + num_labels_cc

    train_data = pd.read_csv(train_data_path, converters={"label": lambda x: np.fromstring(x[1:-1], dtype=float, sep=" ")})
    test_data = pd.read_csv(val_data_path, converters={"label": lambda x: np.fromstring(x[1:-1], dtype=float, sep=" ")})

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
            f"num_labels_pc: {num_labels_pc}\n"
            f"num_labels_cc: {num_labels_cc}\n"
            f"num_labels: {num_labels}\n"
            f"num_epochs: {num_epochs}\n"
            f"num_train_data: {len(train_data)}\n"
            f"num_test_data: {len(test_data)}\n"
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


    class LlamaForMultilabelSequenceClassification(LlamaForSequenceClassification):
        def __init__(self, config):
            super().__init__(config)

        def forward(self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            transformer_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]
            logits = self.score(hidden_states)

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
                else:
                    sequence_lengths = -1

            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                loss_fct_pc = CrossEntropyLoss()
                loss_fct_cc = CrossEntropyLoss()
                
                logits_pc = pooled_logits[:, :num_labels_pc]
                labels_onehot_pc = labels[:, :num_labels_pc]
                labels_pc = torch.argmax(labels_onehot_pc, axis=1)

                logits_cc = pooled_logits[:, num_labels_pc:]
                labels_onehot_cc = labels[:, num_labels_pc:]
                labels_cc = torch.argmax(labels_onehot_cc, axis=1)
                
                loss_pc = loss_fct_pc(logits_pc, labels_pc)
                loss_cc = loss_fct_cc(logits_cc, labels_cc)
                loss = loss_pc + 0.5*loss_cc
            if not return_dict:
                output = (pooled_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )


    model = LlamaForMultilabelSequenceClassification.from_pretrained(
        base_model, 
        num_labels=num_labels, 
        load_in_8bit=True, 
        problem_type="multi_label_classification",
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

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=None
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

    train_data = Dataset.from_pandas(train_data)
    test_data = Dataset.from_pandas(test_data)

    # train_data= train_data.shard(num_shards=5000, index=0)
    # test_data= test_data.shard(num_shards=5000, index=0)

    tokenized_train = train_data.map(preprocess_function, batched=True).remove_columns(["text"]).rename_column("label", "labels")
    tokenized_test = test_data.map(preprocess_function, batched=True).remove_columns(["text"]).rename_column("label", "labels")

    # default is padding to longest
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics_multi(eval_pred):
        predictions, labels = eval_pred
        return cls_metrics_multi(y_pred=predictions, y=labels)

    # Other hyperparameters to consider here is gradient_accumulation_steps, weight decay, learning rate, adam etype  
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
        compute_metrics=compute_metrics_multi,
        )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)