# Local Evaluation Models

This directory stores local embedding models used by evaluation scripts.

The default semantic similarity model is:

```
models/paraphrase-multilingual-MiniLM-L12-v2
```

Download it from Hugging Face:

```
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir models/paraphrase-multilingual-MiniLM-L12-v2
```
