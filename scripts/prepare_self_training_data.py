#!/usr/bin/env python
"""Assemble self-training dataset from generated splits.

Reads the split JSON files from the self-training generation,
filters for correctness using the polaris ground truth,
applies Qwen chat template, and saves as training parquet.
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List

import datasets
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_generation", "src"))
from rewards import rllm_reward_fn_math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", default="data_generation/data/Q3-8B-131072-SFT-1st_self_training")
    parser.add_argument("--polaris-parquet", default="data_generation/data/polaris-data-53K.parquet")
    parser.add_argument("--output-dir", default="threadweaver_sft/data/self_training_17k")
    parser.add_argument("--qwen-model", default="threadweaver_sft/Qwen/Qwen3-8B-131072")
    parser.add_argument("--total-splits", type=int, default=16)
    parser.add_argument("--instruction", default="Let's think step by step and output the final answer within \\boxed{}.")
    args = parser.parse_args()

    polaris_df = pd.read_parquet(args.polaris_parquet)
    total_rows = len(polaris_df)
    split_size = math.ceil(total_rows / args.total_splits)
    print(f"Polaris: {total_rows} rows, split_size={split_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)

    all_records: List[Dict] = []

    for split_idx in range(args.total_splits):
        json_path = os.path.join(args.gen_dir, f"polaris-data-53K_1_split{split_idx}_of_{args.total_splits}.json")
        if not os.path.exists(json_path):
            print(f"Skip split {split_idx}: {json_path} not found")
            continue

        with open(json_path) as f:
            data = json.load(f)

        start = split_idx * split_size
        end = min(start + split_size, total_rows)
        df_slice = polaris_df.iloc[start:end].reset_index(drop=True)

        if len(data) != len(df_slice):
            print(f"WARNING: split {split_idx} has {len(data)} samples but df slice has {len(df_slice)} rows")
            min_len = min(len(data), len(df_slice))
            data = data[:min_len]
            df_slice = df_slice.iloc[:min_len]

        correct_count = 0
        for i, (item, (_, row)) in enumerate(zip(data, df_slice.iterrows())):
            response = item[0]
            ground_truth = row["reward_model"]["ground_truth"]

            if not rllm_reward_fn_math("polaris", response, ground_truth):
                continue
            correct_count += 1

            question = row["prompt"][0]["content"]
            messages = [
                {"role": "user", "content": f"{question} {args.instruction}"},
                {"role": "assistant", "content": response},
            ]
            qwen_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            reasoning = response.split("</think>")[0].strip().removeprefix("<think>")
            content = response.split("</think>")[-1].strip()

            all_records.append({
                "question": question,
                "response_reasoning": reasoning,
                "response_content": content,
                "correctness": True,
                "qwen_text": qwen_text,
                "num_qwen_tokens": len(tokenizer(qwen_text)["input_ids"]),
                "raw_messages": messages,
            })

        print(f"Split {split_idx}: {correct_count}/{len(data)} correct ({100*correct_count/len(data):.1f}%)")

    print(f"\nTotal correct samples: {len(all_records)}")

    os.makedirs(args.output_dir, exist_ok=True)
    ds = datasets.Dataset.from_list(all_records)
    out_path = os.path.join(args.output_dir, "train.parquet")
    ds.to_parquet(out_path)
    print(f"Saved {len(all_records)} samples to {out_path}")


if __name__ == "__main__":
    main()
