Token Entropy Evaluation

This script samples 32 items from the GSM8K test parquet, generates responses up to 1024 tokens using Qwen/Qwen3-4B via vLLM or transformers, and logs per-token entropy per sample to a dedicated Weights & Biases project.

How to run (example):

- With your project environment activated:

```
python token_entropy_eval.py \
  --model "/hpc2hdd/home/zli404/.cache/modelscope/hub/models/Qwen/Qwen3-4B" \
  --data "/hpc2hdd/home/zli404/workspace/verl/data/gsm8k/test.parquet" \
  --project "verl_token_entropy_eval" \
  --experiment "qwen3_4b_gsm8k_token_entropy" \
  --num-samples 32 \
  --max-new-tokens 1024
```

Artifacts:
- Logs per-sample per-token entropy arrays and summary metrics to WandB.
