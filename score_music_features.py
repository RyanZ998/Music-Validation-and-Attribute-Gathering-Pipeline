#!/usr/bin/env python3
"""
Music-Therapy Feature + Evidence Scorer (Enhanced Research Version)
-------------------------------------------------------------------
Reads master_ready_to_score.csv, fills missing evidence tiers with defaults,
computes feature-level scores, evidence-weighted total scores, and outputs
scored_tracks_sorted.csv sorted from highest to lowest Total_score.
"""

import pandas as pd
import numpy as np

# ---------- Base Config ----------
INPUT_FILE = "master_ready_to_score.csv"
OUTPUT_FILE = "scored_tracks_sorted.csv"

BASE_WEIGHTS = {
    "BPM": 0.25,
    "Mode": 0.20,
    "Lyric sentiment valence": 0.25,
    "Lyric sentiment arousal": 0.30,
}

EVIDENCE_MULTIPLIERS = {
    "rct": 1.15,
    "meta": 1.10,
    "systematic": 1.10,
    "observational": 0.95,
    "clinical": 0.95,
    "theoretical": 0.85,
    "mechanistic": 0.85,
    "anecdotal": 0.75,
    "indirect": 0.75,
}

DEFAULT_EVIDENCE = {
    "BPM": "meta",
    "Mode": "theoretical",
    "Lyric sentiment valence": "observational",
    "Lyric sentiment arousal": "observational",
}

LETTER_CUTOFFS = {
    "A+": 97, "A": 93, "A-": 90,
    "B+": 87, "B": 83, "B-": 80,
    "C+": 77, "C": 73, "C-": 70,
    "D": 60, "F": 0,
}

# ---------- Scoring helpers ----------
def range_decay(x, ideal_min, ideal_max, hard_min, hard_max):
    if pd.isna(x):
        return np.nan
    if ideal_min <= x <= ideal_max:
        return 1.0
    if x < ideal_min:
        if x <= hard_min:
            return 0.0
        return (x - hard_min) / (ideal_min - hard_min)
    if x > ideal_max:
        if x >= hard_max:
            return 0.0
        return 1 - (x - ideal_max) / (hard_max - ideal_max)
    return 0.0


def score_bpm(bpm):
    s1 = range_decay(bpm, 60, 80, 50, 130)
    s2 = range_decay(bpm, 100, 120, 50, 130)
    if np.isnan(s1) and np.isnan(s2):
        return np.nan
    return max(s1, s2)


def score_mode(mode):
    if pd.isna(mode):
        return np.nan
    m = str(mode).strip().lower()
    mapping = {"major": 1.0, "mixolydian": 0.8, "dorian": 0.5, "minor": 0.4}
    return mapping.get(m, 0.3)


def score_valence(v):
    return range_decay(v, 0.2, 0.6, -0.5, 1.0)


def score_arousal(a):
    return range_decay(a, 0.2, 0.6, -0.3, 1.0)


def evidence_mult(label):
    if pd.isna(label):
        return 1.0
    key = str(label).lower().strip()
    return EVIDENCE_MULTIPLIERS.get(key, 1.0)


def letter_grade(score):
    if pd.isna(score):
        return "N/A"
    pct = 100 * score
    for g, t in sorted(LETTER_CUTOFFS.items(), key=lambda kv: -kv[1]):
        if pct >= t:
            return g
    return "F"


# ---------- Row scoring ----------
def score_row(row):
    feats = ["BPM", "Mode", "Lyric sentiment valence", "Lyric sentiment arousal"]
    evidence_cols = {
        "BPM": "BPM_evidence",
        "Mode": "Mode_evidence",
        "Lyric sentiment valence": "LyricVal_evidence",
        "Lyric sentiment arousal": "LyricAro_evidence",
    }

    # Compute feature scores
    feature_scores = {
        "BPM": score_bpm(row.get("BPM", np.nan)),
        "Mode": score_mode(row.get("Mode", np.nan)),
        "Lyric sentiment valence": score_valence(row.get("Lyric sentiment valence", np.nan)),
        "Lyric sentiment arousal": score_arousal(row.get("Lyric sentiment arousal", np.nan)),
    }

    # Effective weights (base × evidence multiplier)
    eff_w = {}
    for f in feats:
        label = row.get(evidence_cols[f], np.nan)
        mult = evidence_mult(label)
        eff_w[f] = BASE_WEIGHTS[f] * mult

    # Normalize weights to sum = 1
    total_w = sum(eff_w.values())
    eff_w = {k: v / total_w for k, v in eff_w.items()}

    # Weighted score
    numer = sum(feature_scores[f] * eff_w[f] for f in feats if not np.isnan(feature_scores[f]))
    denom = sum(eff_w[f] for f in feats if not np.isnan(feature_scores[f]))
    total = numer / denom if denom else np.nan
    return feature_scores, eff_w, total


# ---------- Main ----------
def main():
    df = pd.read_csv(INPUT_FILE)

    # Ensure evidence columns exist and fill with defaults if missing
    for f, default in DEFAULT_EVIDENCE.items():
        colname = {
            "BPM": "BPM_evidence",
            "Mode": "Mode_evidence",
            "Lyric sentiment valence": "LyricVal_evidence",
            "Lyric sentiment arousal": "LyricAro_evidence",
        }[f]
        if colname not in df.columns:
            df[colname] = default
        else:
            df[colname] = df[colname].fillna(default)
            df.loc[df[colname].astype(str).str.strip() == "", colname] = default

    results = []
    for _, r in df.iterrows():
        scores, wts, total = score_row(r)
        out = {
            **r.to_dict(),
            "BPM_score": scores["BPM"],
            "Mode_score": scores["Mode"],
            "LyricVal_score": scores["Lyric sentiment valence"],
            "LyricAro_score": scores["Lyric sentiment arousal"],
            "BPM_weight": wts["BPM"],
            "Mode_weight": wts["Mode"],
            "LyricVal_weight": wts["Lyric sentiment valence"],
            "LyricAro_weight": wts["Lyric sentiment arousal"],
            "Total_score": total,
            "Letter_grade": letter_grade(total),
        }
        results.append(out)

    out_df = pd.DataFrame(results)
    out_df.sort_values(by="Total_score", ascending=False, inplace=True)
    out_df.reset_index(drop=True, inplace=True)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done! {len(out_df)} tracks scored & sorted -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
