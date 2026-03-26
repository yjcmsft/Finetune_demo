# Finetune Demo — GPT-OSS 120B & 20B for Financial Fraud Detection

Fine-tune large open-source GPT models (120B and 20B parameter class) on financial fraud-detection data using LoRA/PEFT + DeepSpeed, deployed on Azure ML and served via AKS + vLLM.

## Datasets

| # | Dataset | Type | Source |
|---|---------|------|--------|
| 1 | **ULB Credit Card Fraud (2013)** | Real (European cardholders) | `mlg-ulb/creditcardfraud` on Kaggle |
| 2 | **IEEE-CIS Fraud Detection (2019)** | Real (e-commerce, card-not-present) | `ieee-fraud-detection` Kaggle competition |
| 3 | **Fraudulent E-Commerce Transactions** | Synthetic | `shriyashjagtap/fraudulent-e-commerce-transactions` on Kaggle |
| 4 | **PaySim Mobile Money Transfers** | Synthetic | `ealaxi/paysim1` on Kaggle |
| 5 | **Sparkov Credit Card Transactions** | Synthetic (public domain) | Included in repo (`fraudTrain.csv` / `fraudTest.csv`) |

All datasets are converted into a unified JSONL format (`{"prompt": "...", "completion": "..."}`) with a 90/10 train/validation split.

## Project Structure

```
Finetune_demo/
├── LICENSE
├── README.md
├── training.jsonl
└── gpt120b-finance/
    ├── environment.yml                  # Conda env (PyTorch, Transformers, PEFT, DeepSpeed)
    ├── train_120b_peft.py               # LoRA fine-tuning script for 120B model
    ├── train_20b_peft.py                # LoRA fine-tuning script for 20B model
    ├── ds_config_zerostage3.json        # DeepSpeed ZeRO-3 config (120B)
    ├── ds_config_zerostage2.json        # DeepSpeed ZeRO-2 config (20B)
    ├── aml_job.yaml                     # Azure ML job definition (120B, 8 nodes)
    ├── aml_job_20b.yaml                 # Azure ML job definition (20B, 2 nodes)
    ├── data/
    │   ├── prepare_all_datasets.py      # Download & convert all 5 datasets to JSONL
    │   ├── csv_to_jsonl.py              # Legacy single-CSV converter (Sparkov)
    │   ├── fraudTrain.csv               # Sparkov training data
    │   ├── fraudTest.csv                # Sparkov test data
    │   ├── fraudTrain_fixed.csv
    │   ├── fraudTest_fixed.csv
    │   ├── training.jsonl               # Generated — unified training set
    │   └── validation.jsonl             # Generated — unified validation set
    └── serving/
        ├── adapter_config.json          # LoRA adapter config (120B)
        ├── adapter_config_20b.json      # LoRA adapter config (20B)
        ├── aks-deploy.yaml              # AKS deployment manifest (120B, 8 GPUs)
        └── aks-deploy-20b.yaml          # AKS deployment manifest (20B, 2 GPUs)
```

## Quick Start

### 1. Prepare Data

```bash
# Install dependencies
pip install kaggle pandas scikit-learn

# Set Kaggle credentials
export KAGGLE_USERNAME=<your-username>
export KAGGLE_KEY=<your-api-key>

# Download all 5 datasets and convert to JSONL
cd gpt120b-finance/data
python prepare_all_datasets.py

# Or skip download if CSVs are already in data/raw/
python prepare_all_datasets.py --skip-download

# Cap samples per dataset for quick tests
python prepare_all_datasets.py --skip-download --max-per-dataset 5000
```

### 2. Fine-Tune

#### GPT-OSS 120B (8 nodes, ZeRO-3, LoRA r=32)

```bash
# Local (multi-GPU)
deepspeed --num_gpus 8 train_120b_peft.py \
  --model_id <org/gpt-neox-120b> \
  --train_jsonl data/training.jsonl \
  --val_jsonl data/validation.jsonl

# Azure ML
az ml job create -f aml_job.yaml
```

#### GPT-OSS 20B (2 nodes, ZeRO-2, LoRA r=16)

```bash
# Local (multi-GPU)
deepspeed --num_gpus 4 train_20b_peft.py \
  --model_id <org/gpt-neox-20b> \
  --train_jsonl data/training.jsonl \
  --val_jsonl data/validation.jsonl

# Azure ML
az ml job create -f aml_job_20b.yaml
```

### 3. Deploy on AKS

```bash
# 120B model (8× GPU, 120 Gi memory)
kubectl apply -f serving/aks-deploy.yaml

# 20B model (2× GPU, 48 Gi memory)
kubectl apply -f serving/aks-deploy-20b.yaml
```

## Model Comparison

| Setting | 120B | 20B |
|---------|------|-----|
| DeepSpeed stage | ZeRO-3 | ZeRO-2 |
| LoRA rank (r) | 32 | 16 |
| LoRA alpha | 64 | 32 |
| Batch size / GPU | 1 | 2 |
| Gradient accumulation | 64 | 16 |
| Learning rate | 1e-4 | 2e-4 |
| Epochs | 2 | 3 |
| AML nodes | 8 | 2 |
| Serving GPUs | 8 | 2 |
| Tensor parallelism | 8 | 2 |

## Environment

The Conda environment (`environment.yml`) includes:

- Python 3.10, PyTorch 2.1, Transformers 4.36, PEFT 0.7, DeepSpeed 0.12
- bitsandbytes for QLoRA quantization
- TRL for optional RLHF / DPO training
- Azure AI ML SDK for job submission

## License

See [LICENSE](LICENSE).