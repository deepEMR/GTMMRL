# PyTorch implementation of "Graph and Text Multi-Modal Representation Learning with Momentum Distillation on Electronic Health Records"


## Usage
+ Code for MIMIC-III pre-processing in `preprocessing`
+ Code for running experiments in `gtmmrl/run_gtmmrl.py` 
  + Please log in to [W&B](https://wandb.ai) for logging results.
+ Pre-trained models in `gtmmrl/pretrained_models/Pre/`


## pre-processing
+ step 1: `preprocessing/mimic_db_create/process_mimic_db.py`
+ step 2: `preprocessing/mimic_prepare/extract_text.ipynb`
+ step 3: `preprocessing/mimic_prepare/table2triple.ipynb`
+ step 4: `preprocessing/mimic_prepare/triple2subgraph.ipynb`


