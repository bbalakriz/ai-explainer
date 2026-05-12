#!/usr/bin/env python3
"""
Chat SDG Pipeline

Generates synthetic training data for Thai telecom intent classification
using the sdg_hub framework. Reads existing seed data from JSON files and
generates 20 new examples per intent.

Usage:
    # Using sdg_hub flow (requires sdg_hub installed):
    python run.py --mode flow --model gemini/gemma-4-31b-it

    # Using standalone litellm (lighter dependency):
    python run.py --mode standalone --model anthropic/claude-sonnet-4-20250514

    # Specify output directory:
    python run.py --mode standalone --model openai/gpt-4o --output ../generated

    # Use a local model via Ollama:
    python run.py --mode standalone --model ollama/gemma2 --api-base http://localhost:11434
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Intent descriptions – loaded from external JSON so that run.py and the
# notebook always stay in sync.
# ---------------------------------------------------------------------------
_INTENT_DESC_PATH = Path(__file__).parent / "intent_descriptions.json"
with open(_INTENT_DESC_PATH, encoding="utf-8") as _f:
    INTENT_DESCRIPTIONS: dict[str, str] = json.load(_f)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_seed_data(seed_file: Path) -> dict[str, list[dict]]:
    """Load seed examples from training_dataset.json grouped by intent."""
    with open(seed_file, encoding="utf-8") as fh:
        data = json.load(fh)
    seeds: dict[str, list[dict]] = {}
    for ex in data:
        intent = ex.get("intent", "")
        if intent:
            seeds.setdefault(intent, []).append(ex)
    return seeds


def select_seed_examples(examples: list[dict], max_seeds: int = 5) -> list[dict]:
    """Pick a diverse subset of seed examples (varied history lengths)."""
    by_len = {}
    for ex in examples:
        h_len = len(ex.get("session_history", []))
        by_len.setdefault(h_len, []).append(ex)

    selected = []
    # Take one from each history-length bucket, round-robin
    buckets = sorted(by_len.keys())
    idx = 0
    while len(selected) < max_seeds and idx < len(examples):
        for b in buckets:
            if by_len[b] and len(selected) < max_seeds:
                selected.append(by_len[b].pop(0))
        idx += 1

    # Truncate seed histories to max 5 messages (sliding window)
    for ex in selected:
        if len(ex.get("session_history", [])) > 5:
            ex["session_history"] = ex["session_history"][-5:]

    return selected


def build_prompt(intent: str, seed_examples: list[dict], num_examples: int = 20) -> list[dict]:
    """Build chat messages for the LLM call."""
    prompt_path = Path(__file__).parent / "prompts" / "generate_telco_chat.yaml"

    # Read the YAML template manually (avoid heavy deps)
    import yaml
    with open(prompt_path) as f:
        template_messages = yaml.safe_load(f)

    # Simple Jinja2-style substitution
    seed_json = json.dumps(seed_examples, ensure_ascii=False, indent=2)
    description = INTENT_DESCRIPTIONS.get(intent, f"Intent: {intent}")

    messages = []
    for msg in template_messages:
        content = msg["content"]
        content = content.replace("{{ intent }}", intent)
        content = content.replace("{{ intent_description }}", description)
        content = content.replace("{{ seed_examples }}", seed_json)
        content = content.replace("{{ num_examples }}", str(num_examples))
        content = content.replace("{{ num_seed }}", str(len(seed_examples)))
        messages.append({"role": msg["role"], "content": content})

    return messages


def parse_json_output(text: str) -> list[dict]:
    """Extract a JSON array from the LLM output text."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        json_str = text[start : end + 1]
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing line by line for individual JSON objects
    print("  WARNING: Could not parse full JSON array, attempting recovery...")
    return []


def validate_examples(examples: list[dict], intent: str) -> list[dict]:
    """Validate and fix generated examples."""
    valid = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        # Ensure required fields
        if "user_message" not in ex:
            continue
        ex["intent"] = intent  # Force correct intent
        if "session_history" not in ex:
            ex["session_history"] = []
        # Validate session_history structure
        if not isinstance(ex["session_history"], list):
            ex["session_history"] = []
        cleaned_history = []
        for msg in ex["session_history"]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] in ("user", "assistant"):
                    cleaned_history.append({
                        "role": msg["role"],
                        "content": str(msg["content"]),
                    })
        # Enforce sliding window: keep only the last 5 messages
        if len(cleaned_history) > 5:
            cleaned_history = cleaned_history[-5:]
        ex["session_history"] = cleaned_history
        valid.append({
            "user_message": str(ex["user_message"]),
            "intent": intent,
            "session_history": ex["session_history"],
        })
    return valid


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------

def generate_with_litellm(
    intent: str,
    seed_examples: list[dict],
    num_examples: int,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """Generate examples using litellm directly."""
    import litellm

    messages = build_prompt(intent, seed_examples, num_examples)

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 16384,
        "temperature": 0.8,
        "timeout": 1800,  # 30 min for local models
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    response = litellm.completion(**kwargs)
    text = response.choices[0].message.content
    return parse_json_output(text)


def generate_with_flow(
    intents_data: list[dict],
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Generate using sdg_hub Flow (all intents in one batch)."""
    import pandas as pd
    from sdg_hub import Flow

    flow_path = Path(__file__).parent / "flow.yaml"
    flow = Flow.from_yaml(str(flow_path))

    model_config = {"model": model}
    if api_base:
        model_config["api_base"] = api_base
    if api_key:
        model_config["api_key"] = api_key
    flow.set_model_config(**model_config)

    df = pd.DataFrame(intents_data)
    result = flow.generate(df, max_concurrency=5)

    outputs = {}
    for _, row in result.iterrows():
        intent = row["intent"]
        content = row.get("extract_output_content", "")
        outputs[intent] = content
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Telco Chat SDG Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "flow"],
        default="standalone",
        help="standalone = litellm only; flow = full sdg_hub Flow",
    )
    parser.add_argument(
        "--model",
        default="gemini/gemma-4-31b-it",
        help="LLM model in litellm format (default: gemini/gemma-4-31b-it)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for self-hosted models",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (or set via env: GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)",
    )
    parser.add_argument(
        "--output",
        default="generated_intents",
        help="Output directory (default: ../generated_intents)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=20,
        help="Number of examples to generate per intent (default: 20)",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=5,
        help="Max seed examples to include in prompt (default: 5)",
    )
    parser.add_argument(
        "--intents",
        nargs="*",
        default=None,
        help="Specific intents to generate (default: all)",
    )
    args = parser.parse_args()

    # Paths
    seed_file = Path(__file__).parent.parent / "samples" / "training_dataset.json"
    output_dir = Path(args.output) if args.output else Path(__file__).parent.parent / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load seed data
    print("Loading seed data...")
    seeds = load_seed_data(seed_file)
    print(f"  Found {len(seeds)} intents: {', '.join(seeds.keys())}")

    # Filter intents if specified
    intents_to_generate = args.intents if args.intents else list(seeds.keys())
    intents_to_generate = [i for i in intents_to_generate if i in seeds]

    if not intents_to_generate:
        print("ERROR: No valid intents found to generate.")
        sys.exit(1)

    print(f"\nGenerating {args.num_examples} examples each for {len(intents_to_generate)} intents")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}\n")

    # -----------------------------------------------------------------------
    # Mode: standalone (litellm)
    # -----------------------------------------------------------------------
    if args.mode == "standalone":
        all_results = {}
        for i, intent in enumerate(intents_to_generate, 1):
            print(f"[{i}/{len(intents_to_generate)}] Generating: {intent}")
            seed_subset = select_seed_examples(seeds[intent], args.max_seeds)
            print(f"  Using {len(seed_subset)} seed examples")

            start = time.time()
            try:
                generated = generate_with_litellm(
                    intent=intent,
                    seed_examples=seed_subset,
                    num_examples=args.num_examples,
                    model=args.model,
                    api_base=args.api_base,
                    api_key=args.api_key,
                )
                validated = validate_examples(generated, intent)
                elapsed = time.time() - start
                print(f"  Generated {len(validated)} valid examples ({elapsed:.1f}s)")

                if len(validated) < args.num_examples:
                    print(f"  WARNING: Only {len(validated)}/{args.num_examples} valid examples")

                all_results[intent] = validated

                # Save individual intent file
                out_file = output_dir / f"{intent}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(validated, f, ensure_ascii=False, indent=2)
                print(f"  Saved to {out_file}")

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[intent] = []

    # -----------------------------------------------------------------------
    # Mode: flow (sdg_hub)
    # -----------------------------------------------------------------------
    elif args.mode == "flow":
        try:
            import pandas as pd  
            from sdg_hub import Flow 
        except ImportError:
            print("ERROR: sdg_hub not installed. Run: pip install sdg-hub")
            print("       Or use --mode standalone for litellm-only mode.")
            sys.exit(1)

        # Prepare input DataFrame
        intents_data = []
        for intent in intents_to_generate:
            seed_subset = select_seed_examples(seeds[intent], args.max_seeds)
            intents_data.append({
                "intent": intent,
                "intent_description": INTENT_DESCRIPTIONS.get(intent, intent),
                "seed_examples": json.dumps(seed_subset, ensure_ascii=False, indent=2),
                "num_examples": str(args.num_examples),
                "num_seed": str(len(seed_subset)),
            })

        print("Running sdg_hub flow...")
        start = time.time()
        try:
            outputs = generate_with_flow(
                intents_data=intents_data,
                model=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
            )
            elapsed = time.time() - start
            print(f"Flow completed in {elapsed:.1f}s")

            for intent, raw_text in outputs.items():
                generated = parse_json_output(raw_text)
                validated = validate_examples(generated, intent)
                print(f"  {intent}: {len(validated)} valid examples")

                out_file = output_dir / f"{intent}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(validated, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"ERROR: Flow execution failed: {e}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    total = 0
    for f in sorted(output_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
            total += len(data)
            print(f"  {f.name}: {len(data)} examples")
    print(f"\n  Total: {total} examples in {output_dir}")


if __name__ == "__main__":
    main()
