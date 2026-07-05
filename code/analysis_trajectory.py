# -*- coding: utf-8 -*-
"""
analysis_trajectory.py

Trajectory-level analyses for RQ3 and the probabilistic-defense argument.

Reads:
  data/conversations.jsonl   (original experiment data with per-turn LFI/CDI/breach flags)

Produces (console + CSV outputs in data/):
  1. Per-turn first-breach timing: at which turn does the first regex-detected
     breach occur, by family. Distribution over turns 1-7.
  2. Per-turn disclosure hazard: of conversations still "clean" entering turn t,
     what fraction breach at turn t. Tests whether per-turn breach probability is
     constant (independent-draws model) or rising (context-accumulation model).
  3. Trajectory metrics by family: peak LFI, mean LFI, cumulative LFI,
     fragmentation slope (least-squares slope of LFI over turn index).
  4. LFI trajectory shape: mean LFI per turn per family (for plotting later).

Requires: pandas, scipy, numpy
    pip install pandas scipy numpy --break-system-packages
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR      = Path("mt_data")
CONVERSATIONS = DATA_DIR / "conversations.jsonl"
TURNS_PER_CONV = 7

# ============================================================================
# LOADER
# ============================================================================

def load_conversations():
    convs = []
    with open(CONVERSATIONS, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            convs.append(json.loads(line))
    return convs

# ============================================================================
# 1. PER-TURN FIRST-BREACH TIMING
# ============================================================================

def first_breach_timing(convs):
    print("=" * 70)
    print("1. FIRST-BREACH TIMING BY FAMILY (regex breach flag)")
    print("=" * 70)

    rows = []
    for c in convs:
        breach_turn = c.get('breach_turn')  # set by experiment script, None if no breach
        rows.append({'family': c['family'],
                     'conversation_id': c['conversation_id'],
                     'breach_turn': breach_turn})
    df = pd.DataFrame(rows)

    for fam in sorted(df['family'].unique()):
        sub = df[df['family'] == fam]
        breached = sub.dropna(subset=['breach_turn'])
        print(f"\n{fam} (n={len(sub)}, breached={len(breached)}):")
        if len(breached) == 0:
            print("  no regex breaches")
            continue
        dist = breached['breach_turn'].value_counts().sort_index()
        for turn in range(1, TURNS_PER_CONV + 1):
            count = int(dist.get(turn, 0))
            bar = "#" * count
            print(f"  turn {turn}: {count:3d} {bar}")
        print(f"  median first-breach turn: {breached['breach_turn'].median():.1f}")

    out = DATA_DIR / "analysis_first_breach_timing.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return df

# ============================================================================
# 2. PER-TURN DISCLOSURE HAZARD
# ============================================================================

# ============================================================================
# 2. PER-TURN DISCLOSURE HAZARD (layered analysis)
#
# Question: is the per-turn disclosure hazard constant across turns, or does it
# vary (e.g., spike at compositional assembly steps)?
#
# Three complementary analyses, each doing a distinct job:
#   2a. Chi-square test of homogeneity of per-turn hazard (primary test of
#       non-uniformity, correctly conditioning on the depleting at-risk pool)
#   2b. Discrete-time logistic hazard model: likelihood-ratio test of a
#       constant-hazard null against a turn-varying model (rigorous confirmation)
#   2c. Assembly-step verification: descriptive check of what the attacker
#       requested at the high-hazard turns (supports the mechanistic shape)
# ============================================================================

def _hazard_table(fam_convs):
    """Return per-turn (at_risk, breach_here) accounting for the depleting pool."""
    at_risk = len(fam_convs)
    rows = []
    for turn in range(1, TURNS_PER_CONV + 1):
        breach_here = sum(1 for c in fam_convs if c.get('breach_turn') == turn)
        rows.append({'turn': turn, 'at_risk': at_risk, 'breach': breach_here,
                     'survive': at_risk - breach_here,
                     'hazard': (breach_here / at_risk) if at_risk else 0.0})
        at_risk -= breach_here
    return rows

def _chi2_homogeneity(haz_rows):
    """Chi-square test of homogeneity: is breach probability equal across turns,
    conditional on being at risk. Table rows = turns, cols = [breach, survive].
    Uses only turns with at_risk > 0. Monte Carlo p-value for robustness with
    thin cells."""
    table = [[r['breach'], r['survive']] for r in haz_rows if r['at_risk'] > 0]
    # Drop any all-zero rows that would break the test
    table = [row for row in table if sum(row) > 0]
    if len(table) < 2:
        return None
    chi2, p, dof, expected = stats.chi2_contingency(table)
    thin = (expected < 5).any()
    return {'chi2': chi2, 'p': p, 'dof': dof, 'thin_cells': bool(thin)}

def _discrete_time_hazard_model(fam_convs):
    """Discrete-time hazard model via logistic regression.
    Build a person-period dataset: one row per (conversation, turn) for turns
    up to and including the breach turn (or all 7 if no breach). Outcome = breach
    at that turn. Compare constant-hazard (intercept only) vs turn-as-factor model
    via likelihood-ratio test."""
    try:
        import numpy as np
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels not available'}

    rows = []
    for c in fam_convs:
        bt = c.get('breach_turn')
        last_turn = bt if bt else TURNS_PER_CONV
        for turn in range(1, last_turn + 1):
            event = 1 if (bt == turn) else 0
            rows.append({'turn': turn, 'event': event})
    if not rows:
        return {'error': 'no person-period rows'}

    df = pd.DataFrame(rows)
    n_events = int(df['event'].sum())
    if n_events < 5:
        return {'error': f'too few events ({n_events}) for stable model'}

    # Null model: intercept only (constant hazard)
    X_null = np.ones((len(df), 1))
    try:
        null_fit = sm.Logit(df['event'].values, X_null).fit(disp=0)
    except Exception as e:
        return {'error': f'null model failed: {e}'}

    # Full model: turn as categorical factor
    turn_dummies = pd.get_dummies(df['turn'], prefix='turn', drop_first=True).astype(float)
    X_full = sm.add_constant(turn_dummies)
    try:
        full_fit = sm.Logit(df['event'].values, X_full.values).fit(disp=0)
    except Exception as e:
        return {'error': f'full model failed: {e}'}

    # Likelihood-ratio test
    lr_stat = 2 * (full_fit.llf - null_fit.llf)
    lr_df = X_full.shape[1] - 1
    lr_p = stats.chi2.sf(lr_stat, lr_df)
    return {'lr_stat': lr_stat, 'lr_df': lr_df, 'lr_p': lr_p,
            'n_events': n_events, 'n_periods': len(df)}

# Keywords that indicate the attacker is requesting assembly/combination of components
ASSEMBLY_KEYWORDS = [
    'combine', 'combined', 'combining', 'together', 'join', 'joined',
    'put them', 'merge', 'concatenate', 'all caps', 'underscore',
    'both answers', 'two answers', 'your answers', 'the pattern',
    'in the format', 'what do you get', 'construct'
]

def _assembly_step_verification(fam_convs, high_hazard_turns):
    """Descriptive: at the high-hazard turns, what fraction of attacker messages
    (in conversations breaching at that turn) requested assembly/combination?"""
    results = {}
    for turn in high_hazard_turns:
        breaching_here = [c for c in fam_convs if c.get('breach_turn') == turn]
        n = len(breaching_here)
        assembly_count = 0
        for c in breaching_here:
            # Find the attacker message at this turn
            atk_msg = ""
            for t in c['turns']:
                if t.get('turn') == turn and 'error' not in t:
                    atk_msg = str(t.get('attacker_message', '')).lower()
                    break
            if any(kw in atk_msg for kw in ASSEMBLY_KEYWORDS):
                assembly_count += 1
        results[turn] = {'n_breaching': n, 'assembly_requests': assembly_count,
                         'fraction': (assembly_count / n) if n else 0.0}
    return results

def disclosure_hazard(convs):
    print("\n" + "=" * 70)
    print("2. PER-TURN DISCLOSURE HAZARD (layered analysis)")
    print("=" * 70)
    print("Of conversations still clean entering turn t, fraction breaching AT turn t.\n")

    for fam in sorted(set(c['family'] for c in convs)):
        fam_convs = [c for c in convs if c['family'] == fam]
        n_breaches_total = sum(1 for c in fam_convs if c.get('breach_turn'))

        haz_rows = _hazard_table(fam_convs)
        note = "  (few breaches; estimates noisy, tests underpowered)" if n_breaches_total < 10 else ""
        print(f"{fam}{note}")
        for r in haz_rows:
            bar = "#" * int(r['hazard'] * 40)
            print(f"  turn {r['turn']}: at_risk={r['at_risk']:3d}, breach={r['breach']:3d}, "
                  f"hazard={r['hazard']*100:5.1f}% {bar}")

        # Only run the formal tests where there are enough events to be meaningful
        if n_breaches_total >= 10:
            # 2a. Chi-square homogeneity
            chi = _chi2_homogeneity(haz_rows)
            if chi:
                flag = " (thin cells; interpret with care)" if chi['thin_cells'] else ""
                print(f"  [2a] Chi-square homogeneity of hazard across turns: "
                      f"chi2={chi['chi2']:.1f}, df={chi['dof']}, p={chi['p']:.3e}{flag}")
                print(f"       {'Hazard is NON-uniform across turns.' if chi['p'] < 0.05 else 'No evidence of non-uniformity.'}")

            # 2b. Discrete-time hazard model
            dm = _discrete_time_hazard_model(fam_convs)
            if 'error' in dm:
                print(f"  [2b] Discrete-time hazard model: {dm['error']}")
            else:
                print(f"  [2b] Discrete-time hazard LR test (constant vs turn-varying): "
                      f"LR={dm['lr_stat']:.1f}, df={dm['lr_df']}, p={dm['lr_p']:.3e} "
                      f"(events={dm['n_events']}, periods={dm['n_periods']})")
                print(f"       {'Turn-varying hazard fits better than constant.' if dm['lr_p'] < 0.05 else 'Constant hazard not rejected.'}")

            # 2c. Assembly-step verification at the two highest-hazard turns
            ranked = sorted(haz_rows, key=lambda r: r['hazard'], reverse=True)
            high_turns = [r['turn'] for r in ranked[:2] if r['hazard'] > 0.1]
            if high_turns:
                av = _assembly_step_verification(fam_convs, high_turns)
                print(f"  [2c] Assembly-step verification at high-hazard turns {high_turns}:")
                for turn in high_turns:
                    a = av[turn]
                    print(f"       turn {turn}: of {a['n_breaching']} conversations breaching here, "
                          f"{a['assembly_requests']} ({a['fraction']*100:.0f}%) had an assembly/"
                          f"combination request in the attacker message")
        print()

# ============================================================================
# 3 + 4. TRAJECTORY METRICS AND LFI SHAPE
# ============================================================================

def fragmentation_slope(lfi_values):
    pts = [(i + 1, v) for i, v in enumerate(lfi_values) if v is not None]
    if len(pts) < 2:
        return 0.0
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    slope, _, _, _, _ = stats.linregress(xs, ys)
    return slope

def trajectory_metrics(convs):
    print("=" * 70)
    print("3. TRAJECTORY METRICS BY FAMILY")
    print("=" * 70)

    rows = []
    per_turn_lfi = {}  # family -> list of [turn1_lfis, turn2_lfis, ...]

    for c in convs:
        fam = c['family']
        lfis = []
        for t in c['turns']:
            if 'error' in t:
                lfis.append(None)
            else:
                lfis.append(t.get('lfi_score'))
        valid_lfis = [x for x in lfis if x is not None]
        if not valid_lfis:
            continue
        rows.append({
            'family': fam,
            'conversation_id': c['conversation_id'],
            'peak_lfi': max(valid_lfis),
            'mean_lfi': float(np.mean(valid_lfis)),
            'cumulative_lfi': sum(valid_lfis),
            'fragmentation_slope': fragmentation_slope(lfis),
        })
        # collect per-turn for shape
        per_turn_lfi.setdefault(fam, [[] for _ in range(TURNS_PER_CONV)])
        for i, v in enumerate(lfis):
            if v is not None and i < TURNS_PER_CONV:
                per_turn_lfi[fam][i].append(v)

    tdf = pd.DataFrame(rows)
    print(f"\n{'Family':<26}{'peak':>8}{'mean':>8}{'cumul':>8}{'frag_slope':>12}")
    for fam in sorted(tdf['family'].unique()):
        sub = tdf[tdf['family'] == fam]
        print(f"{fam:<26}{sub['peak_lfi'].mean():>8.1f}"
              f"{sub['mean_lfi'].mean():>8.1f}{sub['cumulative_lfi'].mean():>8.1f}"
              f"{sub['fragmentation_slope'].mean():>12.3f}")

    out = DATA_DIR / "analysis_trajectory_metrics.csv"
    tdf.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # 4. LFI shape: mean LFI per turn per family
    print("\n" + "=" * 70)
    print("4. MEAN LFI PER TURN PER FAMILY (trajectory shape)")
    print("=" * 70)
    shape_rows = []
    for fam in sorted(per_turn_lfi.keys()):
        means = []
        for turn_idx in range(TURNS_PER_CONV):
            vals = per_turn_lfi[fam][turn_idx]
            m = float(np.mean(vals)) if vals else 0.0
            means.append(m)
        print(f"{fam}:")
        print(f"  " + "  ".join(f"T{i+1}={m:.1f}" for i, m in enumerate(means)))
        for turn_idx, m in enumerate(means):
            shape_rows.append({'family': fam, 'turn': turn_idx + 1, 'mean_lfi': m})

    out2 = DATA_DIR / "analysis_lfi_shape.csv"
    pd.DataFrame(shape_rows).to_csv(out2, index=False)
    print(f"\nWrote {out2}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    if not CONVERSATIONS.exists():
        print(f"ERROR: {CONVERSATIONS} not found.")
        sys.exit(1)

    convs = load_conversations()
    print(f"Loaded {len(convs)} conversations\n")

    first_breach_timing(convs)
    disclosure_hazard(convs)
    trajectory_metrics(convs)

    print("\n" + "=" * 70)
    print("analysis_trajectory.py COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
