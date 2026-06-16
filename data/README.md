# Dataset

This directory stores the project dataset files.

## `sft_sql.sqlite`

`sft_sql.sqlite` is the SQLite dataset used for supervised fine-tuning (SFT) data preparation in this project.

Source: https://huggingface.co/datasets/capstone-group/Capstone-dataset/blob/main/sft_sql.sqlite

To download it from the project root:

```
wget -O data/sft_sql.sqlite https://huggingface.co/datasets/capstone-group/Capstone-dataset/resolve/main/sft_sql.sqlite
```

## `final_result.parquet`

`final_result.parquet` is the processed project dataset used to generate SFT and DPO JSONL files.

Source: https://huggingface.co/datasets/capstone-group/Capstone-dataset/blob/main/final_result.parquet

To download it from the project root:

```
wget -O data/final_result.parquet https://huggingface.co/datasets/capstone-group/Capstone-dataset/resolve/main/final_result.parquet
```

## `sft_eval.jsonl`

`sft_eval.jsonl` is the evaluation dataset used for model validation and offline evaluation.

Source: https://huggingface.co/datasets/capstone-group/Capstone-dataset/blob/main/sft_eval.jsonl

To download it from the project root:

```
wget -O data/sft_eval.jsonl https://huggingface.co/datasets/capstone-group/Capstone-dataset/resolve/main/sft_eval.jsonl
```
