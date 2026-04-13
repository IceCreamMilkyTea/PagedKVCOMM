"""Re-evaluate per-node accuracy from experiment logs using LLM-based answer extraction.

Parses [AGENT OUTPUT] lines from a log file, uses the same LLM (Llama-3.1-8B-Instruct)
to extract predicted values, and computes per-node accuracy broken down by CRS applied
vs. not applied.

Usage:
    python -m KVCOMM.scripts.evaluate_node_accuracy \
        --log_path logs/hf_crs_ablation_rtx_pro_10939241.log \
        --llm_name /path/to/Llama-3.1-8B-Instruct \
        --dataset datasets/gsm8k/gsm8k.jsonl \
        --output_path KVCOMM/result/node_accuracy_eval.json
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.gsm8k_dataset import gsm_get_predict


def parse_log(log_path: str):
    """Parse log file to extract per-batch agent outputs and CRS status."""
    with open(log_path) as f:
        lines = f.readlines()

    batch_idx = -1
    batches = {}  # batch_idx -> {agent_id: text}
    batch_has_crs = {}

    for line in lines:
        if "[BATCH]" in line:
            m = re.search(r"\[BATCH\]\s*(\d+)", line)
            if m:
                batch_idx = int(m.group(1))
                batches[batch_idx] = {}
                batch_has_crs[batch_idx] = False

        if "[CRS:hf] APPLIED" in line and batch_idx >= 0:
            batch_has_crs[batch_idx] = True

        if "[AGENT OUTPUT]" in line and batch_idx >= 0:
            idx = line.index("{")
            data = json.loads(line[idx:])
            aid = data["agent_id"]
            text = data["text"]
            batches[batch_idx][aid] = text

    return batches, batch_has_crs


def load_ground_truth(dataset_path: str):
    """Load ground truth answers from GSM8K dataset."""
    gt_answers = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            ans = data.get("answer", "")
            if "####" in ans:
                ans = ans.split("####")[-1].strip().replace(",", "")
            gt_answers.append(ans)
    return gt_answers


async def extract_answer_with_model(text: str, tokenizer, model, device) -> str:
    """Use LLM to extract answer from agent output text."""
    answer = text.split("A: ", 1)[-1] if "A: " in text else text

    messages = [
        {
            "role": "system",
            "content": (
                "You are professional in extracting the exact value from the user response. "
                "Given the following user response, you should only output the value that is argued "
                "as the most correct answer by the user for evaluation, i.e., only an integer or "
                "float-type value in one line."
            ),
        },
        {"role": "user", "content": answer},
    ]

    token_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {
        "input_ids": token_inputs.to(device),
        "attention_mask": torch.ones_like(token_inputs).to(device),
    }
    prompt_length = inputs["input_ids"].shape[-1]
    generate_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "return_dict_in_generate": True,
        "return_legacy_cache": False,
    }

    outputs = await asyncio.to_thread(model.generate, **inputs, **generate_kwargs)
    generated = outputs.sequences[:, prompt_length:]
    response_message = tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    return gsm_get_predict(response_message)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True, nargs="+",
                        help="Log file path(s) to evaluate")
    parser.add_argument("--llm_name", required=True,
                        help="Path to Llama model")
    parser.add_argument("--dataset", default="datasets/gsm8k/gsm8k.jsonl",
                        help="Path to GSM8K dataset")
    parser.add_argument("--output_path", default="KVCOMM/result/node_accuracy_eval.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.llm_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Load ground truth
    gt_answers = load_ground_truth(args.dataset)
    print(f"Loaded {len(gt_answers)} ground truth answers")

    role_map = {"0": "FinalRefer", "1": "MathSolver", "2": "MathAnalyst", "3": "Programmer"}
    all_results = {}

    for log_path in args.log_path:
        print(f"\n{'='*60}")
        print(f"Evaluating: {log_path}")
        print(f"{'='*60}")

        batches, batch_has_crs = parse_log(log_path)
        n_batches = len(batches)
        print(f"Parsed {n_batches} batches, {sum(batch_has_crs.values())} with CRS applied")

        # Evaluate each agent output with model
        # node_results[aid][batch_idx] = {"pred": str, "correct": bool}
        node_results = {str(i): {} for i in range(4)}

        for batch_i in sorted(batches.keys()):
            if batch_i >= len(gt_answers):
                break
            gt = gt_answers[batch_i]

            for aid, text in batches[batch_i].items():
                pred = await extract_answer_with_model(text, tokenizer, model, device)
                try:
                    correct = float(pred) == float(gt)
                except Exception:
                    correct = False
                node_results[aid][batch_i] = {"pred": pred, "correct": correct}

            if (batch_i + 1) % 100 == 0:
                print(f"  Evaluated {batch_i + 1}/{n_batches} batches")

        # Compute accuracy
        log_name = Path(log_path).stem
        log_result = {"log": log_path, "per_node": {}, "crs_breakdown": {}}

        for aid in ["1", "2", "3", "0"]:
            role = role_map[aid]
            results = node_results[aid]
            total = len(results)
            correct = sum(1 for r in results.values() if r["correct"])
            acc = correct / total if total > 0 else 0

            # CRS breakdown
            crs_correct = crs_total = no_crs_correct = no_crs_total = 0
            for batch_i, r in results.items():
                if batch_has_crs.get(batch_i, False):
                    crs_total += 1
                    if r["correct"]:
                        crs_correct += 1
                else:
                    no_crs_total += 1
                    if r["correct"]:
                        no_crs_correct += 1

            crs_acc = crs_correct / crs_total if crs_total > 0 else None
            no_crs_acc = no_crs_correct / no_crs_total if no_crs_total > 0 else None

            log_result["per_node"][role] = {
                "node_id": aid,
                "accuracy": round(acc * 100, 2),
                "correct": correct,
                "total": total,
            }
            log_result["crs_breakdown"][role] = {
                "crs_applied": {
                    "accuracy": round(crs_acc * 100, 2) if crs_acc is not None else None,
                    "correct": crs_correct,
                    "total": crs_total,
                },
                "no_crs": {
                    "accuracy": round(no_crs_acc * 100, 2) if no_crs_acc is not None else None,
                    "correct": no_crs_correct,
                    "total": no_crs_total,
                },
                "delta": round((crs_acc - no_crs_acc) * 100, 2) if crs_acc is not None and no_crs_acc is not None else None,
            }

            print(f"\n  Node {aid} ({role}): {correct}/{total} = {acc*100:.2f}%")
            if crs_total > 0:
                print(f"    CRS Applied: {crs_correct}/{crs_total} = {crs_acc*100:.2f}%")
                print(f"    No CRS:      {no_crs_correct}/{no_crs_total} = {no_crs_acc*100:.2f}%")
                print(f"    Delta:       {(crs_acc - no_crs_acc)*100:+.2f}")

        all_results[log_name] = log_result

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
