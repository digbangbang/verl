import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Args:
    model: str
    data: str
    project: str
    experiment: str
    num_samples: int = 32
    max_new_tokens: int = 1024
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() else "float32"
    temperature: float = 0.7
    top_p: float = 0.95
    trust_remote_code: bool = True
    csv_dir: Optional[str] = None
    batch_size: int = 8
    attn_implementation: Optional[str] = None  # e.g., "flash_attention_2"


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Model path or HF repo id")
    p.add_argument("--data", type=str, required=True, help="Path to parquet test file")
    p.add_argument("--project", type=str, default="verl_token_entropy_eval")
    p.add_argument("--experiment", type=str, default="qwen3_4b_gsm8k_token_entropy")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--dtype", type=str, default=None, choices=[None, "float32", "float16", "bfloat16"])  # type: ignore
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--csv-dir", type=str, default=None, help="Directory to save per-sample CSVs")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for generation to improve GPU usage")
    p.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        choices=[None, "flash_attention_2", "sdpa", "eager"],  # type: ignore
        help="Attention backend hint for transformers (requires support)",
    )
    args = p.parse_args()

    A = Args(
        model=args.model,
        data=args.data,
        project=args.project,
        experiment=args.experiment,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype if args.dtype is not None else ("bfloat16" if torch.cuda.is_available() else "float32"),
        device=args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
        trust_remote_code=args.trust_remote_code,
    csv_dir=args.csv_dir,
    batch_size=args.batch_size,
    attn_implementation=args.attn_implementation,
    )
    return A


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_dataset(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    if len(df) <= k:
        return df
    return df.sample(n=k, random_state=seed).reset_index(drop=True)


def load_data(parquet_path: str) -> List[str]:
    df = pd.read_parquet(parquet_path)
    # Try typical columns
    for col in ["prompt", "question", "input", "instruction", "text"]:
        if col in df.columns:
            prompts = df[col].astype(str).tolist()
            break
    else:
        # fallback: first string-like column
        for c in df.columns:
            if df[c].dtype == object or np.issubdtype(df[c].dtype, np.str_):
                prompts = df[c].astype(str).tolist()
                break
        else:
            raise ValueError(f"No suitable text column found in {parquet_path}; columns={df.columns.tolist()}")
    return prompts


def build_model_and_tokenizer(model_path: str, device: str, dtype: str, trust_remote_code: bool, attn_implementation: Optional[str] = None):
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    extra_kwargs = {}
    if attn_implementation is not None:
        extra_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=trust_remote_code,
        **extra_kwargs,
    )
    model.eval()
    return model, tok


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V]
    probs = torch.softmax(logits, dim=-1)
    # numerical stability: clamp
    probs = probs.clamp_min(1e-12)
    ent = -(probs * probs.log()).sum(dim=-1)  # [B, T]
    return ent


def generate_and_entropy(args: Args):
    set_seed(args.seed)

    # Load prompts
    prompts = load_data(args.data)
    prompts = sample_dataset(pd.DataFrame({"prompt": prompts}), args.num_samples, args.seed)["prompt"].tolist()

    # Throughput-friendly settings
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # speed on Ampere+
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Load model
    model, tok = build_model_and_tokenizer(
        args.model, args.device, args.dtype, args.trust_remote_code, args.attn_implementation
    )

    # Init wandb
    wandb.init(project=args.project, name=args.experiment, config={
        "model": args.model,
        "data": args.data,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "dtype": args.dtype,
    })

    per_sample_summaries = []

    idx_global = 0
    bs = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(prompts), bs), desc="eval", total=(len(prompts) + bs - 1) // bs):
        batch_prompts = prompts[start : start + bs]
        # Batched tokenization
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
        input_lengths = (inputs.input_ids != tok.pad_token_id).sum(dim=1)

        with torch.inference_mode():
            gen_out = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        scores = gen_out.scores
        if scores is None or len(scores) == 0:
            continue
        logits = torch.stack(scores, dim=0).permute(1, 0, 2)  # [B, T_new, V]
        ents = compute_token_entropy(logits).tolist()  # list of [T_new] per sample

        # Per-sample processing in the batch
        for bi in range(len(batch_prompts)):
            ent = ents[bi]
            inp_len = int(input_lengths[bi].item())
            gen_ids = gen_out.sequences[bi][inp_len:]
            generated_text = tok.decode(gen_ids, skip_special_tokens=True)
            token_strs = tok.convert_ids_to_tokens(gen_ids.tolist())
            min_len = min(len(ent), len(token_strs))
            ent = ent[:min_len]
            token_strs = token_strs[:min_len]

            # Log per-sample entropy line chart
            xs = [list(range(len(ent)))]
            ys = [ent]
            line_plot = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=[f"sample_{idx_global}"],
                title=f"Token Entropy (sample {idx_global})",
                xname="index",
            )

            wandb.log({
                "sample/index": idx_global,
                "sample/prompt": batch_prompts[bi],
                "sample/response": generated_text,
                "sample/token_entropy_plot": line_plot,
                "sample/entropy_mean": float(np.mean(ent)) if len(ent) else 0.0,
                "sample/entropy_max": float(np.max(ent)) if len(ent) else 0.0,
                "sample/entropy_min": float(np.min(ent)) if len(ent) else 0.0,
                "sample/tokens": len(ent),
            })

            per_sample_summaries.append({
                "idx": idx_global,
                "entropy_mean": float(np.mean(ent)) if len(ent) else 0.0,
                "entropy_max": float(np.max(ent)) if len(ent) else 0.0,
                "entropy_min": float(np.min(ent)) if len(ent) else 0.0,
                "tokens": len(ent),
            })

            # Save per-sample CSV
            out_dir = args.csv_dir or os.path.join(os.getcwd(), "token_entropy_samples")
            os.makedirs(out_dir, exist_ok=True)
            df_sample = pd.DataFrame({
                "index": list(range(len(ent))),
                "token": token_strs,
                "entropy": [float(x) for x in ent],
            })
            csv_path = os.path.join(out_dir, f"sample_{idx_global:04d}.csv")
            df_sample.to_csv(csv_path, index=False)
            idx_global += 1
            print(f"Saved sample CSV: {csv_path}")

    # Log a summary table
    if per_sample_summaries:
        sum_table = wandb.Table(columns=["idx", "entropy_mean", "entropy_max", "entropy_min", "tokens"])
        for r in per_sample_summaries:
            sum_table.add_data(r["idx"], r["entropy_mean"], r["entropy_max"], r["entropy_min"], r["tokens"])
        wandb.log({"summary/token_entropy": sum_table})

    # No aggregated CSV; per-sample CSVs are saved during the loop.

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    generate_and_entropy(args)
