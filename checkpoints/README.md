# Model Checkpoints

This directory stores local model checkpoints and LoRA adapters.

The default LoRA checkpoint expected by the inference code is:

```
checkpoints/train_2026-05-23-20-15-30
```

The released LoRA checkpoint used for inference can be downloaded here. Training-time LoRA adapters created by LlamaFactory are stored under `LlamaFactory/saves/`, not this directory.

The DPO training config expects the previous SFT LoRA adapter at `LlamaFactory/saves/Qwen3-4B-Instruct-2507/lora`.

Download source:

https://huggingface.co/datasets/capstone-group/Capstone-dataset/tree/main/train_2026-05-23-20-15-30

From the project root, download it with:

```
hf download capstone-group/Capstone-dataset --repo-type dataset --include "train_2026-05-23-20-15-30/*" --local-dir checkpoints
```
