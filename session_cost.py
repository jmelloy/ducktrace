#!/usr/bin/env python3
"""Calculate Claude Code session cost from a JSONL transcript file."""

import json
import sys

# Pricing per million tokens (USD)
PRICING = {
    "claude-sonnet-4-6": {
        "input":          3.00,
        "cache_write_5m": 1.00,
        "cache_write_1h": 3.75,
        "cache_read":     0.30,
        "output":        15.00,
    },
    "claude-haiku-4-5": {
        "input":          0.80,
        "cache_write_5m": 0.08,
        "cache_write_1h": 1.00,
        "cache_read":     0.08,
        "output":         4.00,
    },
}

def calculate_cost(path: str) -> None:
    totals: dict[str, dict] = {}
    seen_msg_ids: set[str] = set()

    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "assistant":
                continue

            msg = r.get("message", {})
            msg_id = msg.get("id")
            if msg_id in seen_msg_ids:
                continue
            seen_msg_ids.add(msg_id)

            model = msg.get("model", "unknown")
            u = msg.get("usage", {})
            cc = u.get("cache_creation", {})

            if model not in totals:
                totals[model] = {
                    "input": 0,
                    "cache_write_5m": 0,
                    "cache_write_1h": 0,
                    "cache_read": 0,
                    "output": 0,
                }

            t = totals[model]
            t["input"]          += u.get("input_tokens", 0)
            t["cache_read"]     += u.get("cache_read_input_tokens", 0)
            t["output"]         += u.get("output_tokens", 0)
            t["cache_write_5m"] += cc.get("ephemeral_5m_input_tokens", 0)
            t["cache_write_1h"] += cc.get("ephemeral_1h_input_tokens", 0)

    grand_total = 0.0

    for model, t in totals.items():
        p = PRICING.get(model)
        if p is None:
            print(f"WARNING: no pricing for {model}, skipping")
            continue

        costs = {k: t[k] / 1e6 * p[k] for k in p}
        model_total = sum(costs.values())
        grand_total += model_total

        print(f"\n{model}")
        print(f"  {'tokens':30s}  {'cost':>10s}")
        print(f"  {'-'*42}")
        for k in p:
            print(f"  {k:30s}  ${costs[k]:>9.4f}  ({t[k]:,} tokens)")
        print(f"  {'TOTAL':30s}  ${model_total:>9.4f}")

    print(f"\n{'='*44}")
    print(f"  {'GRAND TOTAL':30s}  ${grand_total:>9.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <session.jsonl>")
        sys.exit(1)
    calculate_cost(sys.argv[1])
