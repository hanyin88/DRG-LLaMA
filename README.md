# DRG-LLaMA

This repository contains the code used for [DRG-LLaMA : Tuning LLaMA Model to Predict Diagnosis-related Group for Hospitalized Patients](https://arxiv.org/abs/2309.12625) and implementation instructions.

## Local setup
Install dependencies. We used conda environment.
```
conda env create -f environment.yml
```
Activate conda environment.
```
conda activate DRG-LLaMA
```


## MIMIC-IV pre-processing

1) You must have obtained access to MIMIC-IV database: https://physionet.org/content/mimiciv/. 
2) Download "discharge.csv" and "drgcodes.csv" from MIMIC-IV and update "dc_summary_path" in `paths.json` to the file locations. We provided mapping rule file in the data folder ("my_mapping_path").
3) We provided "DRG_34.csv" in the data folder, which is the official DRG v34.0 codes (https://www.cms.gov/icd10m/version34-fullcode-cms/fullcode_cms/P0372.html). 
4) We provided "DRG34_Mapping.csv", which is a mapping rule to unify MS-DRGs over years to a single version -- MS-DRG v34.0. Details of the method can be found in Supplemental Method 1 of the paper.  
5) In your terminal, navigate to the project directory, then type the following commands:
```
python -m data.MIMIC_Preprocessing
```
The script will generate files in "train_set_path", "test_set_path" and "id2label_path". These will be used for single label DRGs prediction.

6) Then run the pre-processing scripts for two-lable DRGs prediction.
```
python -m data.Two_Label_DRG_Preprocessing
```
The script will generate files in "multi_train_set_path", "multi_test_set_path" and "drg_34_dissection_path".



## Running the models
We provided llama_single.py and llama_two.py, which implement fine-tuning of LLaMA with LoRA for the single label and two-label approaches of DRGs prediction, respectively. We largely adopted the framework from https://github.com/tloen/alpaca-lora.

Example usaige:
```
python -m llama_single --base_model 'decapoda-research/llama-7b-hf' --model_size '7b'
```
Hyperparameters can be adjusted such as:
```
python -m llama_single \
    --base_model 'decapoda-research/llama-7b-hf' \
    --model_size '7b' \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --cutoff_len 1024 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
```
clinicalBERT_single.py implements the fine-tuning of clinicalBERT for the single label DRGs prediciton. It can be run as:
```
python -m clinicalBERT_single.py --base_model ""emilyalsentzer/Bio_ClinicalBERT""
```

Please refer to https://github.com/JHLiu7/EarlyDRGPrediction for the implementation of CAML. We adopted evaluation functions in CAML to compute performance metrics (utils/eval_utils.py). The details on the inference of MS-DRG from predicted base DRG and CC/MCC status (funciton 'map_rule' in eval_utils.py) can be found in Supplemental Method 2 of the paper. 
