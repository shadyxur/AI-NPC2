# -*- coding: utf-8 -*-
"""
analysis_vocab.py

Vocabulary mirroring precursor analysis.

Tests the hypothesis from the manual review: do eventually-breaching
conversations show higher attacker-vocabulary overlap in their PRE-breach turns
than conversations that remain clean?

Method:
  For each turn, compute the lexical overlap between the attacker message and the
  NPC response (Jaccard over content-word sets, stopwords removed). For each
  conversation, summarize pre-breach mirroring. Compare breaching vs clean
  conversations.

Reads:
  data/conversations.jsonl
  data/conversations_judged_flat.csv   (for SJI-based breach labels)

Produces (console + CSV):
  1. Per-conversation mean pre-breach mirroring score
  2. Comparison: breaching vs clean conversations (Mann-Whitney U)
  3. Per-turn mirroring trajectory, breaching vs clean

Requires: pandas, scipy, numpy
    pip install pandas scipy numpy --break-system-packages

NOTE: This is an exploratory analysis. Lexical overlap is a coarse proxy for
"vocabulary mirroring." A null result characterizes mirroring as a stable
generative feature; a positive result suggests it is an early-warning signal.
Interpret accordingly and do not overclaim.
"""

import json
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR        = Path("mt_data")
CONVERSATIONS   = DATA_DIR / "conversations.jsonl"
JUDGED_FLAT_CSV = DATA_DIR / "conversations_judged_flat.csv"

# Minimal stopword list. Kept deliberately small and transparent.
STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from as is are was were
be been being it its this that these those i you he she they we me him her them us my
your his their our mine yours hers theirs ours what which who whom whose when where why
how all any both each few more most other some such no nor not only own same so than too
very can will just do does did doing would should could ought im id youre theyre what's
about into over after before above below up down out off again further once here there
""".split())

WORD_RE = re.compile(r"[a-z][a-z'\-]+")

def content_words(text):
    if not text:
        return set()
    words = WORD_RE.findall(str(text).lower())
    return set(w for w in words if w not in STOPWORDS and len(w) > 2)

def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# ============================================================================
# LOADERS
# ============================================================================

def load_conversations():
    convs = {}
    with open(CONVERSATIONS, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            convs[r['conversation_id']] = r
    return convs

def load_breach_labels():
    df = pd.read_csv(JUDGED_FLAT_CSV)
    df['sji_level'] = pd.to_numeric(df['sji_level'], errors='coerce')
    labels = {}
    for _, r in df.iterrows():
        if pd.isna(r['sji_level']):
            continue
        labels[r['conversation_id']] = int(r['sji_level']) >= 1  # True = breached
    return labels

# ============================================================================
# MIRRORING COMPUTATION
# ============================================================================

def compute_mirroring(convs, labels):
    """For each conversation, compute per-turn mirroring and a pre-breach summary."""
    per_conv = []
    per_turn_records = []

    for cid, conv in convs.items():
        if cid not in labels:
            continue
        breached = labels[cid]
        breach_turn = conv.get('breach_turn')  # regex first-breach turn, may be None

        turn_scores = []
        for t in conv['turns']:
            if 'error' in t:
                turn_scores.append(None)
                continue
            atk_words = content_words(t.get('attacker_message', ''))
            npc_words = content_words(t.get('npc_response', ''))
            score = jaccard(atk_words, npc_words)
            turn_scores.append(score)
            per_turn_records.append({
                'conversation_id': cid,
                'family': conv['family'],
                'turn': t['turn'],
                'mirroring': score,
                'breached': breached,
            })

        # Pre-breach window: turns strictly before the regex breach turn.
        # For clean conversations (no breach), use all turns.
        if breach_turn:
            pre = [s for i, s in enumerate(turn_scores)
                   if s is not None and (i + 1) < breach_turn]
        else:
            pre = [s for s in turn_scores if s is not None]

        mean_pre = float(np.mean(pre)) if pre else None

        per_conv.append({
            'conversation_id': cid,
            'family': conv['family'],
            'breached': breached,
            'breach_turn': breach_turn,
            'mean_pre_breach_mirroring': mean_pre,
            'n_pre_turns': len(pre),
        })

    return pd.DataFrame(per_conv), pd.DataFrame(per_turn_records)

# ============================================================================
# ANALYSIS
# ============================================================================

def run_analysis(conv_df, turn_df):
    print("=" * 70)
    print("VOCABULARY MIRRORING PRECURSOR ANALYSIS")
    print("=" * 70)

    valid = conv_df.dropna(subset=['mean_pre_breach_mirroring'])
    breaching = valid[valid['breached']]
    clean     = valid[~valid['breached']]

    print(f"\nConversations with computable pre-breach mirroring: {len(valid)}")
    print(f"  Breaching: {len(breaching)}")
    print(f"  Clean:     {len(clean)}")

    if len(breaching) < 5 or len(clean) < 5:
        print("\n  WARNING: one group is very small; comparison underpowered.")

    b_mean = breaching['mean_pre_breach_mirroring'].mean()
    c_mean = clean['mean_pre_breach_mirroring'].mean()
    print(f"\nMean pre-breach mirroring:")
    print(f"  Breaching conversations: {b_mean:.4f}")
    print(f"  Clean conversations:     {c_mean:.4f}")

    # Mann-Whitney U (non-parametric; mirroring scores are not normal)
    if len(breaching) >= 5 and len(clean) >= 5:
        u, p = stats.mannwhitneyu(
            breaching['mean_pre_breach_mirroring'],
            clean['mean_pre_breach_mirroring'],
            alternative='two-sided'
        )
        print(f"\nMann-Whitney U test:")
        print(f"  U = {u:.1f}, p = {p:.4f}")
        if p < 0.05:
            direction = "higher" if b_mean > c_mean else "lower"
            print(f"  Significant: breaching conversations show {direction} pre-breach mirroring.")
            print(f"  -> mirroring may function as an early-warning signal.")
        else:
            print(f"  Not significant: no evidence that mirroring differs pre-breach.")
            print(f"  -> consistent with mirroring as a stable generative feature.")

    # Per-family breakdown WITH within-family Mann-Whitney tests.
    # This is the correct test of whether mirroring precedes breach: comparing
    # breaching vs clean WITHIN each family avoids the composition effect that
    # makes the pooled comparison misleading (the highest-mirroring family is
    # also the highest-breaching one, so pooling conflates family with mirroring).
    print(f"\nPer-family mean pre-breach mirroring (breaching vs clean) with within-family tests:")
    print(f"  (a within-family test isolates whether mirroring precedes breach,")
    print(f"   free of the cross-family composition effect in the pooled result)")
    for fam in sorted(valid['family'].unique()):
        fb = breaching[breaching['family'] == fam]['mean_pre_breach_mirroring']
        fc = clean[clean['family'] == fam]['mean_pre_breach_mirroring']
        fb_m = fb.mean() if len(fb) else float('nan')
        fc_m = fc.mean() if len(fc) else float('nan')
        line = (f"  {fam:<26} breaching={fb_m:.4f} (n={len(fb)})  "
                f"clean={fc_m:.4f} (n={len(fc)})")
        # Within-family Mann-Whitney requires both groups present and non-trivial
        if len(fb) >= 5 and len(fc) >= 5:
            u, p = stats.mannwhitneyu(fb, fc, alternative='two-sided')
            direction = "breaching higher" if fb_m > fc_m else "clean higher"
            sig = "SIGNIFICANT" if p < 0.05 else "n.s."
            line += f"  | U={u:.0f}, p={p:.3f} ({sig}, {direction})"
        else:
            reason = "no clean cases" if len(fc) == 0 else (
                     "no breaching cases" if len(fb) == 0 else "group too small")
            line += f"  | within-family test not possible ({reason})"
        print(line)
    print(f"\n  Interpretation: if no family shows a significant within-family")
    print(f"  difference, the pooled 'breaching > clean' result is a composition")
    print(f"  effect (driven by which families breach) rather than evidence that")
    print(f"  mirroring precedes breach within comparable conversations.")

    # Per-turn mirroring trajectory
    print(f"\nMean mirroring per turn (breaching vs clean):")
    print(f"{'turn':>5}{'breaching':>12}{'clean':>12}")
    for turn in sorted(turn_df['turn'].unique()):
        tb = turn_df[(turn_df['turn'] == turn) & (turn_df['breached'])]['mirroring']
        tc = turn_df[(turn_df['turn'] == turn) & (~turn_df['breached'])]['mirroring']
        tb_m = tb.mean() if len(tb) else float('nan')
        tc_m = tc.mean() if len(tc) else float('nan')
        print(f"{turn:>5}{tb_m:>12.4f}{tc_m:>12.4f}")

    out1 = DATA_DIR / "analysis_vocab_per_conversation.csv"
    conv_df.to_csv(out1, index=False)
    out2 = DATA_DIR / "analysis_vocab_per_turn.csv"
    turn_df.to_csv(out2, index=False)
    print(f"\nWrote {out1}")
    print(f"Wrote {out2}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    if not CONVERSATIONS.exists():
        print(f"ERROR: {CONVERSATIONS} not found.")
        sys.exit(1)
    if not JUDGED_FLAT_CSV.exists():
        print(f"ERROR: {JUDGED_FLAT_CSV} not found. Run flatten_judge.py first.")
        sys.exit(1)

    convs = load_conversations()
    labels = load_breach_labels()
    conv_df, turn_df = compute_mirroring(convs, labels)
    run_analysis(conv_df, turn_df)

    print("\n" + "=" * 70)
    print("analysis_vocab.py COMPLETE")
    print("Reminder: lexical overlap is a coarse proxy. Interpret cautiously.")
    print("=" * 70)

if __name__ == "__main__":
    main()
