# Multi-Turn Secret Extraction from Persona-Embedded LLM Agents

This repository contains the code and data for **"Multi-Turn Secret Extraction from Persona-Embedded LLM Agents"**. This experiment runs the same target NPC configuration (Optimal) against four attack families across seven-turn conversations, producing 1,000 conversations and 7,000 NPC responses.

The target NPC, Aethelgard the Eternal, guards a secret phrase (AZURE_PHOENIXX) under a persona-embedded system prompt. An automated Gemini 2.5 Flash attacker, conditioned on one of four strategies, attempts extraction across seven turns per conversation. The manuscript's primary disclosure measures are the judge-assigned Semantic Judgment Index (SJI) and a regex-based Component Disclosure Index (CDI), used as a comparison baseline. The Linguistic Fragmentation Index (LFI) is also computed throughout the pipeline but is exploratory: it is not used or discussed in the current manuscript. See "Notes on the data" below.

## Repository structure

```
code/
  full_experi7t.py            Runs the full experiment: generates all 1,000 conversations
  flatten_results.py          Flattens raw JSONL into per-response and per-conversation CSVs
  judge_vali.py               Validates the judge prompt on pilot-scale data (Sonnet 4.6)
  batch_judge.py               Runs the full judge pass via the Anthropic Batch API (Opus 4.7)
  flatten_judge.py            Flattens judge verdicts into a CSV with derived SJI levels
  analysis_trajectory.py      [supplementary] Breach timing, hazard model, LFI trajectory metrics
  analysis_vocab.py           Vocabulary mirroring precursor analysis
  analysis_corrected_timing.py  Corrects breach timing using NPC-response-only detection

data/
  conversations.jsonl          Raw experiment output: one JSON object per conversation
  conversations_judged.jsonl   Raw judge verdicts: one JSON object per conversation

csv/
  conversations_full.csv               Per-response data (7,000 rows), includes LFI columns (supplementary)
  conversations_summary.csv            Per-conversation trajectory metrics (1,000 rows), includes LFI (supplementary)
  conversations_judged_flat.csv        Flattened judge verdicts with SJI levels
  analysis_first_breach_timing.csv     First regex-detected breach turn by conversation
  analysis_trajectory_metrics.csv      [supplementary] Peak, mean, cumulative LFI and fragmentation slope
  analysis_lfi_shape.csv               [supplementary] Mean LFI per turn per family
  analysis_vocab_per_conversation.csv  Pre-breach mirroring scores per conversation
  analysis_vocab_per_turn.csv          Per-turn mirroring scores, breaching vs. clean
  corrected_breach_timing.csv          Corrected first-disclosure turns and hazard results, used in manuscript Section 6.3
```

## Pipeline order

The scripts are meant to be run in this order. Each step's output feeds the next.

1. **full_experi7t.py** generates the raw experiment data. It runs all 1,000 conversations across four attack families (Targeted Semantic Task Decomposition, Structured Data, Information Extraction, Persona Bypass; stored internally as `Task_Decomposition`, `Structured_Data`, `Information_Extraction`, `Persona_Bypass`) against the Optimal-configuration NPC, seven turns each, and writes `data/conversations.jsonl`. The script is resume-safe: rerunning it skips conversation IDs already present in the output file.

2. **flatten_results.py** reads `conversations.jsonl` and writes two CSVs: one row per NPC response, and one row per conversation with trajectory-level metrics (peak LFI, mean LFI, cumulative LFI, fragmentation slope; supplementary, see below). In this repository these are named `conversations_full.csv` and `conversations_summary.csv`.

3. **judge_vali.py** was used during development to validate the judge prompt against small pilot JSONL files before committing to the full run. It calls Claude Sonnet 4.6 directly (not batched) and prints a per-conversation summary for manual inspection.

4. **batch_judge.py** runs the full judge pass over `conversations.jsonl` using the Anthropic Batch API with Claude Opus 4.7. It supports `submit`, `status`, `retrieve`, and a combined `run` command, and writes `data/conversations_judged.jsonl`.

5. **flatten_judge.py** reads the judge output and writes `conversations_judged_flat.csv`, one row per conversation with the judge's disclosure fields flattened and a derived Semantic Judgment Index (SJI, ordinal 0 to 3: clean, partial/structural, distributed, direct). SJI is the manuscript's primary disclosure measure.

6. **analysis_trajectory.py** (supplementary, not used in the current manuscript) reads `conversations.jsonl` and produces first-breach timing by family, a disclosure hazard analysis on the uncorrected regex breach flag, and LFI trajectory metrics and shape by family.

7. **analysis_vocab.py** reads `conversations.jsonl` and `conversations_judged_flat.csv` and tests whether attacker-vocabulary mirroring in pre-breach turns differs between eventually-breaching and clean conversations, both pooled and within family. This is the analysis behind manuscript Section 6.3's mirroring result.

8. **analysis_corrected_timing.py** re-derives breach timing using NPC-response-only detection, correcting a measurement issue in the original `detect_leak()` logic where a partial-match branch could register a breach based on component co-presence across the attacker's message and prior context rather than actual NPC disclosure. It reports the corrected timing distribution, an echo split (attacker-supplied vs. self-generated components), and re-runs the hazard analysis on corrected breach turns. This corrected hazard analysis, not the one in `analysis_trajectory.py`, is what the manuscript reports in Section 6.2. It does not modify `conversations.jsonl` or change SJI-based breach rates, which come from the judge reading NPC responses directly.

## Requirements

```
pandas
numpy
scipy
statsmodels
anthropic
google-genai
python-dotenv
```

Install with:

```
pip install pandas numpy scipy statsmodels anthropic google-genai python-dotenv
```

`full_experi7t.py` requires a `GEMINI_API_KEY` environment variable. `judge_vali.py` and `batch_judge.py` require an `ANTHROPIC_API_KEY`. Both scripts load these from a `.env` file.

## Notes on the data

Attacker and NPC messages in `conversations.jsonl` and `conversations_full.csv` are the literal generated text, including the secret phrase where disclosure occurred. The secret (AZURE_PHOENIXX) is a fictional soul-binding phrase for a fantasy game NPC with no real-world sensitivity; it is published as-is for reproducibility.

The manuscript's primary disclosure measure is the Semantic Judgment Index (SJI), assigned by the Claude Opus 4.7 judge and validated against a human-rated sample. A regex-based Component Disclosure Index (CDI) is used only as a comparison baseline in the instrument-agreement analysis (manuscript Section 6.3).

The Linguistic Fragmentation Index (LFI) and its component scores (U, B, G, N), along with the derived trajectory metrics (peak, mean, cumulative LFI, fragmentation slope) and `analysis_trajectory.py`, are retained in this repository as supplementary, exploratory material from earlier stages of the project. They are not used or discussed in the current manuscript text.
