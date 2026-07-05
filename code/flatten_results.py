# -*- coding: utf-8 -*-
"""
Flatten a conversations JSONL file into two analysis-ready CSVs:

  <name>_responses.csv      -- one row per NPC response
  <name>_conversations.csv  -- one row per conversation, with trajectory metrics

Usage:
    python flatten_results.py [path/to/conversations.jsonl]

Default path: data/conversations.jsonl (the full experiment output).
For pilot:    python flatten_results.py data/pilot2_conversations.jsonl
"""

import json
import csv
import sys
from pathlib import Path

DEFAULT_JSONL = Path("data") / "conversations.jsonl"

RESPONSE_FIELDS = [
    "conversation_id", "family", "conversation_num", "turn",
    "attacker_message", "npc_response", "response_length",
    "lfi_score", "lfi_U", "lfi_B", "lfi_G", "lfi_N",
    "is_leaked", "leak_type", "leak_confidence",
    "cdi_level", "cdi_detail",
    "breach_turn",
    "timestamp",
    "error",
]

CONVERSATION_FIELDS = [
    "conversation_id", "family", "conversation_num",
    "n_turns_completed", "breach_turn",
    "cdi_level", "cdi_detail",
    "peak_lfi", "mean_lfi", "cumulative_lfi", "fragmentation_slope",
    "completed_at",
]

def fragmentation_slope(lfi_values):
    pts = [(i + 1, v) for i, v in enumerate(lfi_values) if v is not None]
    n = len(pts)
    if n < 2:
        return 0.0
    sum_x  = sum(p[0] for p in pts)
    sum_y  = sum(p[1] for p in pts)
    sum_xy = sum(p[0] * p[1] for p in pts)
    sum_x2 = sum(p[0] * p[0] for p in pts)
    denom  = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom

def flatten(jsonl_path):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Cannot find {jsonl_path}.")

    # Derive output CSV paths from the JSONL stem.
    # e.g. "pilot2_conversations.jsonl" -> "pilot2_responses.csv" and "pilot2_conversations.csv"
    stem = jsonl_path.stem
    if stem.endswith("_conversations"):
        base_name = stem[: -len("_conversations")]
    else:
        base_name = stem
    out_dir = jsonl_path.parent
    responses_csv     = out_dir / f"{base_name}_responses.csv"
    conversations_csv = out_dir / f"{base_name}_conversations.csv"

    response_rows     = []
    conversation_rows = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN: skipping malformed line {line_num}: {e}")
                continue

            conv_id      = conv["conversation_id"]
            family       = conv["family"]
            conv_num     = conv["conversation_num"]
            cdi_level    = conv.get("cdi_level")
            cdi_detail   = conv.get("cdi_detail")
            breach_turn  = conv.get("breach_turn")
            completed_at = conv.get("completed_at")

            lfi_per_turn = []

            for turn_rec in conv.get("turns", []):
                row = {
                    "conversation_id":  conv_id,
                    "family":           family,
                    "conversation_num": conv_num,
                    "turn":             turn_rec.get("turn"),
                    "attacker_message": turn_rec.get("attacker_message"),
                    "npc_response":     turn_rec.get("npc_response"),
                    "response_length":  turn_rec.get("response_length"),
                    "lfi_score":        turn_rec.get("lfi_score"),
                    "lfi_U":            turn_rec.get("lfi_U"),
                    "lfi_B":            turn_rec.get("lfi_B"),
                    "lfi_G":            turn_rec.get("lfi_G"),
                    "lfi_N":            turn_rec.get("lfi_N"),
                    "is_leaked":        turn_rec.get("is_leaked"),
                    "leak_type":        turn_rec.get("leak_type"),
                    "leak_confidence":  turn_rec.get("leak_confidence"),
                    "cdi_level":        cdi_level,
                    "cdi_detail":       cdi_detail,
                    "breach_turn":      breach_turn,
                    "timestamp":        turn_rec.get("timestamp"),
                    "error":            turn_rec.get("error"),
                }
                response_rows.append(row)

                if "error" not in turn_rec and turn_rec.get("lfi_score") is not None:
                    lfi_per_turn.append(turn_rec["lfi_score"])

            if lfi_per_turn:
                peak_lfi       = max(lfi_per_turn)
                mean_lfi       = sum(lfi_per_turn) / len(lfi_per_turn)
                cumulative_lfi = sum(lfi_per_turn)
                frag_slope     = fragmentation_slope(lfi_per_turn)
            else:
                peak_lfi = mean_lfi = cumulative_lfi = frag_slope = None

            conversation_rows.append({
                "conversation_id":     conv_id,
                "family":              family,
                "conversation_num":    conv_num,
                "n_turns_completed":   conv.get("n_turns_completed"),
                "breach_turn":         breach_turn,
                "cdi_level":           cdi_level,
                "cdi_detail":          cdi_detail,
                "peak_lfi":            peak_lfi,
                "mean_lfi":            mean_lfi,
                "cumulative_lfi":      cumulative_lfi,
                "fragmentation_slope": frag_slope,
                "completed_at":        completed_at,
            })

    with open(responses_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=RESPONSE_FIELDS)
        w.writeheader()
        w.writerows(response_rows)
    print(f"Wrote {len(response_rows)} rows to {responses_csv}")

    with open(conversations_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CONVERSATION_FIELDS)
        w.writeheader()
        w.writerows(conversation_rows)
    print(f"Wrote {len(conversation_rows)} rows to {conversations_csv}")

    print(f"\n=== SUMMARY ===")
    by_family = {}
    for r in conversation_rows:
        fam = r["family"]
        if fam not in by_family:
            by_family[fam] = {"n": 0, "breaches": 0, "cdi_dist": {0:0, 1:0, 2:0, 3:0}}
        by_family[fam]["n"] += 1
        if r["breach_turn"] is not None:
            by_family[fam]["breaches"] += 1
        if r["cdi_level"] in (0, 1, 2, 3):
            by_family[fam]["cdi_dist"][r["cdi_level"]] += 1

    for fam, stats in by_family.items():
        asr = stats["breaches"] / stats["n"] * 100 if stats["n"] else 0
        print(f"  {fam}")
        print(f"    Conversations: {stats['n']}, Breaches: {stats['breaches']} ({asr:.1f}%)")
        print(f"    CDI distribution: {stats['cdi_dist']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = DEFAULT_JSONL
    flatten(path)
