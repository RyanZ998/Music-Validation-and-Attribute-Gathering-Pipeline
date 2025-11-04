#!/usr/bin/env python3
"""
Lyric Integrity Checker
-----------------------
Scans master_ready_to_score.csv and flags entries with likely incomplete,
missing, or descriptive (summary-like) lyric content.

Outputs: lyrics_integrity_report.csv
"""

import pandas as pd
import numpy as np
import re

INPUT_FILE = "master_ready_to_score.csv"
OUTPUT_FILE = "lyrics_integrity_report.csv"

# Keywords that often indicate descriptive or non-lyric text
SUMMARY_KEYWORDS = [
    "originally written", "composed", "published", "recorded",
    "song has", "as the song", "this song", "track features",
    "performed by", "is about", "theme of", "lyrics talk about",
    "music was", "believed to be", "released in", "instrumental",
]

# Phrases that indicate missing or short content
MISSING_PATTERN = re.compile(r"^\s*$")

def flag_lyrics_quality(lyrics):
    """Return category: GOOD / SHORT / SUMMARY / MISSING / INSTRUMENTAL"""
    if pd.isna(lyrics):
        return "MISSING"

    text = str(lyrics).strip()
    word_count = len(text.split())

    # Missing or empty
    if word_count == 0 or MISSING_PATTERN.match(text):
        return "MISSING"

    # Instrumental or classical tags
    if re.search(r"\b(instrumental|no lyrics|piano|sonata|etude|nocturne|symphony)\b", text.lower()):
        return "INSTRUMENTAL"

    # Very short (e.g., 1-2 sentences only)
    if word_count < 25:
        return "SHORT"

    # Contains summary phrases (not actual lyrics)
    if any(k in text.lower() for k in SUMMARY_KEYWORDS):
        return "SUMMARY"

    # Otherwise fine
    return "GOOD"


def main():
    df = pd.read_csv(INPUT_FILE)
    if "Lyrics" not in df.columns:
        raise ValueError("CSV missing 'Lyrics' column")

    # Apply checker
    df["Lyrics_status"] = df["Lyrics"].apply(flag_lyrics_quality)

    # Aggregate stats
    counts = df["Lyrics_status"].value_counts().to_dict()
    print("Lyric Integrity Summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # Save report of non-GOOD entries
    bad_df = df[df["Lyrics_status"].isin(["SHORT", "SUMMARY", "MISSING", "INSTRUMENTAL"])].copy()
    bad_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n⚠️  {len(bad_df)} tracks flagged -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
