#!/usr/bin/env python3
"""
Telco Chat SDG Pipeline (sdg_hub only)

Generates synthetic Thai language training data for intent classification.
Uses sdg_hub Flow to batch all intents with built in concurrency.

Usage:
    python run_flow.py --model gemini/gemma-4-31b-it
    python run_flow.py --model anthropic/claude-sonnet-4-20250514 --num-examples 50
    python run_flow.py --model ollama/gemma2 --api-base http://localhost:11434
    python run_flow.py --intents MOBILE_BILLING_CHECK_DUE_DATE MOBILE_SIM_ACTIVATION_POSTPAID
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sdg_hub import Flow

PIPELINE_DIR = Path(__file__).parent
INTENT_DESCRIPTIONS: dict[str, str] = json.loads(
    (PIPELINE_DIR / "intent_descriptions.json").read_text(encoding="utf-8")
)


def load_seed_data(seed_file: Path) -> dict[str, list[dict]]:
    """load seed examples from training_dataset.json grouped by intent."""
    data = json.loads(seed_file.read_text(encoding="utf-8"))
    seeds: dict[str, list[dict]] = {}
    for ex in data:
        intent = ex.get("intent", "")
        if intent:
            seeds.setdefault(intent, []).append(ex)
    return seeds


def select_seed_examples(examples: list[dict], max_seeds: int = 5) -> list[dict]:
    """pick a diverse subset of seed examples (varied history lengths)."""
    by_len: dict[int, list[dict]] = {}
    for ex in examples:
        h_len = len(ex.get("session_history", []))
        by_len.setdefault(h_len, []).append(ex)

    selected: list[dict] = []
    buckets = sorted(by_len.keys())
    idx = 0
    while len(selected) < max_seeds and idx < len(examples):
        for b in buckets:
            if by_len[b] and len(selected) < max_seeds:
                selected.append(by_len[b].pop(0))
        idx += 1

    # sliding window: keep at most 5 history messages per seed
    for ex in selected:
        if len(ex.get("session_history", [])) > 5:
            ex["session_history"] = ex["session_history"][-5:]

    return selected


def parse_json_output(text: str) -> list[dict]:
    """extract a JSON array from raw LLM output text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    print("  WARNING: could not parse JSON array, returning empty list")
    return []


def validate_examples(examples: list[dict], intent: str) -> list[dict]:
    """validate and fix generated examples."""
    valid = []
    for ex in examples:
        if not isinstance(ex, dict) or "user_message" not in ex:
            continue

        history = ex.get("session_history", [])
        if not isinstance(history, list):
            history = []

        cleaned = []
        for msg in history:
            if (isinstance(msg, dict)
                    and msg.get("role") in ("user", "assistant")
                    and "content" in msg):
                cleaned.append({"role": msg["role"], "content": str(msg["content"])})

        if len(cleaned) > 5:
            cleaned = cleaned[-5:]

        valid.append({
            "user_message": str(ex["user_message"]),
            "intent": intent,
            "session_history": cleaned,
        })
    return valid


def main():
    parser = argparse.ArgumentParser(description="Telco Chat SDG Pipeline (sdg_hub)")
    parser.add_argument("--model", default="gemini/gemma-4-31b-it",
                        help="LLM model in litellm format")
    parser.add_argument("--api-base", default=None,
                        help="API base URL for self hosted models")
    parser.add_argument("--api-key", default=None,
                        help="API key (or set via env variable)")
    parser.add_argument("--output", default="generated_intents",
                        help="output directory")
    parser.add_argument("--num-examples", type=int, default=20,
                        help="examples to generate per intent")
    parser.add_argument("--max-seeds", type=int, default=5,
                        help="max seed examples in prompt")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="max parallel LLM requests (default: 5, lower if hitting RPM limits)")
    parser.add_argument("--intents", nargs="*", default=None,
                        help="specific intents to generate (default: all)")
    args = parser.parse_args()

    seed_file = PIPELINE_DIR.parent / "samples" / "training_dataset.json"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load seeds and build the input dataframe ---

    print("Loading seed data...")
    seeds = load_seed_data(seed_file)
    print(f"  Found {len(seeds)} intents")

    targets = args.intents or list(seeds.keys())
    targets = [i for i in targets if i in seeds]
    if not targets:
        print("ERROR: no valid intents found.")
        sys.exit(1)

    rows = []
    for intent in targets:
        subset = select_seed_examples(seeds[intent], args.max_seeds)
        rows.append({
            "intent": intent,
            "intent_description": INTENT_DESCRIPTIONS.get(intent, intent),
            "seed_examples": json.dumps(subset, ensure_ascii=False, indent=2),
            "num_examples": str(args.num_examples),
            "num_seed": str(len(subset)),
        })

    df = pd.DataFrame(rows)

    # --- load flow, configure model and run ---

    flow = Flow.from_yaml(str(PIPELINE_DIR / "flow.yaml"))

    model_config = {"model": args.model}
    if args.api_base:
        model_config["api_base"] = args.api_base
    if args.api_key:
        model_config["api_key"] = args.api_key

    model_config["timeout"] = 300.0  
    flow.set_model_config(**model_config)

    # sanity check: verify the LLM block actually received the model name
    for block in flow.blocks:
        if getattr(block, "block_type", None) == "llm":
            actual = getattr(block, "model", None)
            if actual != args.model:
                print(f"ERROR: block '{block.block_name}' has model='{actual}', "
                      f"expected '{args.model}'")
                sys.exit(1)

    print(f"\nGenerating {args.num_examples} examples each for {len(targets)} intents")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {output_dir}\n")

    start = time.time()
    result = flow.generate(df, max_concurrency=args.concurrency)
    elapsed = time.time() - start
    print(f"Flow completed in {elapsed:.1f}s\n")

    # --- post process: parse, validate and save ---

    for _, row in result.iterrows():
        intent = row["intent"]
        raw_text = row.get("extract_output_content", "")
        generated = parse_json_output(raw_text)
        validated = validate_examples(generated, intent)
        print(f"  {intent}: {len(validated)} valid examples")

        out_file = output_dir / f"{intent}.json"
        out_file.write_text(
            json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- summary ---

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    total = 0
    for f in sorted(output_dir.glob("*.json")):
        data = json.loads(f.read_text())
        total += len(data)
        print(f"  {f.name}: {len(data)} examples")
    print(f"\n  Total: {total} examples in {output_dir}")


if __name__ == "__main__":
    main()
