# Capstone-Repose

Recommended on **Ubuntu 22.04** (Python 3.12 + PyTorch 2.8.0 + CUDA 12.8).

Running the Qwen3-4B LoRA model is recommended on a GPU with at least **32 GB VRAM**.

## Project structure
```
Capstone-Repose/
|-- configs/              # Runtime paths and generation settings
|-- data/                 # Downloaded datasets and optional small samples
|-- checkpoints/          # Local LoRA checkpoints
|-- models/               # Local evaluation embedding models
|-- notebooks/            # Data processing and ML development notebooks
|-- outputs/              # Local generated outputs and evaluation results
|-- src/
|   |-- data_prep/        # SFT/DPO data generation
|   |-- evaluation/       # Offline evaluation scripts
|   |-- inference/        # CLI inference
|   `-- web/              # Flask web app
`-- training/             # LlamaFactory dataset and training configs
```

### Clone the repo
```
git clone https://github.com/junhengm1/Capstone-Repose.git
cd Capstone-Repose
```

### Setup environment
```
# Create conda environment
conda create -n Capstone-Repose python=3.12 -y
conda activate Capstone-Repose

# Install PyTorch 2.8.0 (CUDA 12.8)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt
```

### Download the dataset

Download the dataset files from Hugging Face and place them under the `data` directory.

```
mkdir -p data
wget -O data/sft_sql.sqlite https://huggingface.co/datasets/capstone-group/Capstone-dataset/resolve/main/sft_sql.sqlite
wget -O data/final_result.parquet https://huggingface.co/datasets/capstone-group/Capstone-dataset/resolve/main/final_result.parquet
wget -O data/sft_eval.jsonl https://huggingface.co/datasets/capstone-group/Capstone-dataset/resolve/main/sft_eval.jsonl
```

### Download the Qwen model

Download the Qwen model from ModelScope.

```
# Install the ModelScope package
pip install modelscope

# Install Git LFS for downloading large model files
sudo apt update
sudo apt install -y git-lfs
git lfs install

# Clone the Qwen3-4B model repository from ModelScope
git clone https://www.modelscope.cn/Qwen/Qwen3-4B.git

# Install vLLM for efficient model inference and serving
pip install vllm
```

### Notebooks

Exploratory and data-science notebooks are stored under `notebooks/`.

- `notebooks/Capstone_PD_v3.ipynb`: prepares and validates raw geospatial/property data, including fire facilities, bushfire-prone areas, fire history, vegetation, renewable project data, and property-level feature engineering.
- `notebooks/Capstone_ML_v2.ipynb`: trains and evaluates the LightGBM bushfire risk model using the processed features, then generates calibrated risk probability, score, and level outputs for downstream SFT/DPO data preparation.

To work with the notebooks, install the project environment and start Jupyter:

```
pip install -r requirements.txt
jupyter lab
```

## Method 1: Run with the released checkpoint

Use this method if you only want to run inference or the web app with the released fine-tuned LoRA checkpoint.

### Download the fine-tuned checkpoint

The default inference config expects the LoRA checkpoint under `checkpoints/train_2026-05-23-20-15-30`.

```
# Install the Hugging Face CLI if needed
pip install -U huggingface_hub

# Download the fine-tuned checkpoint into the local checkpoints directory
mkdir -p checkpoints
hf download capstone-group/Capstone-dataset --repo-type dataset --include "train_2026-05-23-20-15-30/*" --local-dir checkpoints
```

Checkpoint source:
https://huggingface.co/datasets/capstone-group/Capstone-dataset/tree/main/train_2026-05-23-20-15-30

### Run inference or the web app

Runtime paths are configured in `configs/default.yaml`. By default, the app uses:

- `data/sft_sql.sqlite` for the SQLite lookup database
- `Qwen3-4B` for the base model directory
- `checkpoints/train_2026-05-23-20-15-30` for the LoRA checkpoint

```
# Run terminal inference
python src/inference/run_llm.py

# Run the Flask web app
python src/web/app.py --port 5000
```

You can override paths with environment variables:

```
export CAPSTONE_DB_PATH=data/sft_sql.sqlite
export CAPSTONE_BASE_MODEL=Qwen3-4B
export CAPSTONE_LORA_PATH=checkpoints/train_2026-05-23-20-15-30
```

### Evaluate the model

The evaluation script reads `data/sft_eval.jsonl` by default and writes results to `outputs/eval`.

For semantic similarity metrics, the script uses a local MiniLM sentence embedding model. Download it before running evaluation:

```
# Download the MiniLM semantic evaluation model
mkdir -p models
hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir models/paraphrase-multilingual-MiniLM-L12-v2
```

Run evaluation:

```
# Evaluate the released LoRA checkpoint
python src/evaluation/eval.py --model_variant lora

# Evaluate both the base model and the LoRA checkpoint
python src/evaluation/eval.py --model_variant both
```

Evaluation outputs are saved under:

```
outputs/eval/
```

You can override evaluation paths if needed:

```
python src/evaluation/eval.py \
  --eval-data data/sft_eval.jsonl \
  --output-dir outputs/eval \
  --semantic-model-path models/paraphrase-multilingual-MiniLM-L12-v2
```

## Method 2: Build data and train with LlamaFactory

Use this method if you want to regenerate SFT/DPO JSONL data and train the model yourself.

### Install LlamaFactory

Clone LlamaFactory and install it in editable mode for model fine-tuning and evaluation.

```
# Clone the LlamaFactory repository with shallow history
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory

# Install LlamaFactory in editable mode
pip install -e .

# Install metric-related dependencies
pip install -r requirements/metrics.txt
```

### Prepare LlamaFactory datasets

Generate the SFT and DPO JSONL files, then copy them into the LlamaFactory `data` directory. Also copy the evaluation file `sft_eval.jsonl`.

```
# Go back to the Capstone-Repose project root if you are inside LlamaFactory
cd ..

# Generate the SFT dataset
python src/data_prep/build_sft.py --out outputs/sft_data.jsonl

# Generate the DPO preference dataset
python src/data_prep/build_dpo.py --out outputs/dpo_data.jsonl

# Copy both JSONL files into LlamaFactory's data directory
cp outputs/sft_data.jsonl LlamaFactory/data/sft_data.jsonl
cp outputs/dpo_data.jsonl LlamaFactory/data/dpo_data.jsonl
cp data/sft_eval.jsonl LlamaFactory/data/sft_eval.jsonl
```

The same dataset registration is also stored in `training/dataset_info.json`. Copy these entries into `LlamaFactory/data/dataset_info.json`.

```
{
  "sft_data": {
    "file_name": "sft_data.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "dpo_data": {
    "file_name": "dpo_data.jsonl",
    "formatting": "alpaca",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  },
  "sft_eval": {
    "file_name": "sft_eval.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

The SFT and DPO training configs are stored under `training/llamafactory/`. They use project-relative paths from inside the `LlamaFactory` directory.

### Open the LlamaFactory Web UI

Start the LlamaFactory Web UI after installation.

```
# Disable the online version check
export DISABLE_VERSION_CHECK=1

# Launch the Web UI (in LlamaFactory dir)
llamafactory-cli webui
```

When starting SFT training in the Web UI, select the `sft_data` dataset and use `training/llamafactory/qwen3_lora_sft.yaml` as the reference configuration.

When starting DPO training in the Web UI, select the `dpo_data` dataset and use `training/llamafactory/qwen3_lora_dpo.yaml` as the reference configuration. The DPO run should load the previous SFT checkpoint from `LlamaFactory/saves`.

Run SFT first:

```
# Example: run SFT training from the Capstone-Repose project root
cd LlamaFactory
llamafactory-cli train ../training/llamafactory/qwen3_lora_sft.yaml
cd ..
```

Before DPO training, use the previous SFT LoRA adapter saved by LlamaFactory under:

```
LlamaFactory/saves/Qwen3-4B-Instruct-2507/lora
```

The SFT config writes that adapter to this path by default.

Then run DPO:

```
# Example: run DPO training from the Capstone-Repose project root
cd LlamaFactory
llamafactory-cli train ../training/llamafactory/qwen3_lora_dpo.yaml
cd ..
```

Important training settings in `training/llamafactory/qwen3_lora_sft.yaml`:

```
stage: sft
dataset: sft_data
dataset_dir: data
model_name_or_path: ../Qwen3-4B
output_dir: saves/Qwen3-4B-Instruct-2507/lora/train_2026-05-19-19-00-50
template: qwen3_nothink
enable_thinking: true
finetuning_type: lora
learning_rate: 5.0e-05
num_train_epochs: 1.0
```

Important training settings in `training/llamafactory/qwen3_lora_dpo.yaml`:

```
stage: dpo
dataset: dpo_data
dataset_dir: data
adapter_name_or_path: saves/Qwen3-4B-Instruct-2507/lora/train_2026-05-19-19-00-50
model_name_or_path: ../Qwen3-4B
output_dir: ../checkpoints/train_2026-05-23-20-15-30
template: qwen3_nothink
finetuning_type: lora
pref_loss: sigmoid
pref_beta: 0.7
pref_ftx: 0.1
use_swanlab: false
```

## License

This project is proprietary work prepared for Onyx. See `LICENSE` for details.
