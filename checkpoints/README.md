# Model Checkpoints

This directory stores local model checkpoints and LoRA adapters.

The default LoRA checkpoint expected by the inference code is:

```
checkpoints/train_2026-05-23-20-15-30
```

The DPO training config expects the previous SFT LoRA adapter at:

```
checkpoints/train_2026-05-19-19-00-50
```

Download source:

https://huggingface.co/datasets/capstone-group/Capstone-dataset/tree/main/train_2026-05-23-20-15-30

From the project root, download it with:

```
huggingface-cli download capstone-group/Capstone-dataset --repo-type dataset --include "train_2026-05-23-20-15-30/*" --local-dir checkpoints
```
