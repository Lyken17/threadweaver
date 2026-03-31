#!/usr/bin/env python
"""Prepare RL training data from Polaris-53K for veRL.

Creates a parquet with columns expected by veRL's RL dataset:
- prompt: list of chat message dicts [{"role": "user", "content": ...}]
- data_source: string identifier
- reward_model: dict with {"answer": ground_truth_answer}
"""
import argparse
import os

import datasets
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polaris-parquet", default="data_generation/data/polaris-data-53K.parquet")
    parser.add_argument("--output-dir", default="threadweaver_sft/data/polaris_rl")
    parser.add_argument("--instruction", default="Let's think step by step and output the final answer within \\boxed{}.")
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.polaris_parquet)
    print(f"Loaded {len(df)} rows from Polaris-53K")

    records = []
    for _, row in df.iterrows():
        question = row["prompt"][0]["content"]
        ground_truth = row["reward_model"]["ground_truth"]

        records.append({
            "prompt": [{"role": "user", "content": f"{question} {args.instruction}"}],
            "data_source": "polaris",
            "reward_model": {"answer": ground_truth, "ground_truth": ground_truth},
        })

    ds = datasets.Dataset.from_list(records)
    ds = ds.shuffle(seed=args.seed)

    train_ds = ds.select(range(len(ds) - args.val_size))
    val_ds = ds.select(range(len(ds) - args.val_size, len(ds)))

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(args.output_dir, "val.parquet"))

    print(f"Saved {len(train_ds)} train + {len(val_ds)} val to {args.output_dir}")


if __name__ == "__main__":
    main()
