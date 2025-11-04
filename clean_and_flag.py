import pandas as pd
import re
from datetime import date
import csv  # add this at top

INFILE = "step2_lyrics_output.csv"   
OUT_READY = "master_ready_to_score.csv"
OUT_NEEDS = "master_needs_fill.csv"


def fix_mojibake(s: str):
    """Repair typical UTF-8→Latin-1 mojibake only when detected."""
    if not isinstance(s, str) or s == "":
        return s
    # common telltales: Ã, Â, â€™, â€œ, â€, etc.
    if any(mark in s for mark in ("Ã", "Â", "â€™", "â€œ", "â€\x9d", "â€“", "â€”")):
        try:
            return s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return s
    return s



def clean_lyrics(s: str):
    if not isinstance(s, str): return s
    s = re.sub(r"(?is)\b\d+\s+Contributors.*?$", "", s)
    s = re.sub(r"(?is)Read More.*?$", "", s)
    s = re.sub(r"(?is)Page \d+(?:\s+Page \d+)*", "", s)
    return s.strip()

df = pd.read_csv(INFILE)


# Ensure columns exist
must_cols = ["Title","Artist","Track ID","Source Link","BPM","Mode",
             "Lyric sentiment valence","Lyric sentiment arousal","Lyrics",
             "BPM Source","Mode Source","Curator","Date added"]
for c in must_cols:
    if c not in df.columns:
        df[c] = None

# AFTER reading the CSV and after you fill *_new, normalize Title/Artist once:
for col in ["Title", "Artist"]:
    if col in df.columns:
        df[col] = df[col].apply(fix_mojibake)

# Fix mojibake (also pick *_new if present)
for col in ["Title","Artist","Title_new","Artist_new"]:
    if col in df.columns:
        df[col] = df[col].apply(fix_mojibake)

if "Title_new" in df.columns:
    df["Title"] = df["Title_new"].where(df["Title_new"].notna() & (df["Title_new"].str.strip()!=""), df["Title"])
if "Artist_new" in df.columns:
    df["Artist"] = df["Artist_new"].where(df["Artist_new"].notna() & (df["Artist_new"].str.strip()!=""), df["Artist"])

# Clean lyrics and coerce numerics
df["Lyrics"] = df["Lyrics"].apply(clean_lyrics)
for numcol in ["BPM","Lyric sentiment valence","Lyric sentiment arousal"]:
    df[numcol] = pd.to_numeric(df[numcol], errors="coerce")
    if numcol != "BPM":
        df.loc[df[numcol]==0, numcol] = pd.NA  # drop bogus zeros

# Set provenance if values exist
df.loc[df["BPM"].notna() & df["BPM Source"].isna(), "BPM Source"] = "spotify/analysis"
df.loc[df["Mode"].notna() & df["Mode Source"].isna(), "Mode Source"] = "spotify/analysis"

# Minimal readiness rule
df["required_ok"] = (
    df["Track ID"].notna() &
    df["Source Link"].notna() &
    df["BPM"].notna() &
    df["Mode"].notna() &
    df["Lyric sentiment valence"].notna()
)

def missing_list(row):
    miss = []
    for c in ["Track ID","Source Link","BPM","Mode","Lyric sentiment valence"]:
        v = row.get(c)
        if pd.isna(v) or (isinstance(v, str) and not v.strip()):
            miss.append(c)
    return ", ".join(miss)

df["missing"] = df.apply(missing_list, axis=1)
df["ready_to_score"] = df["required_ok"].map({True:"READY", False:"HOLD"})

# Stamp date if empty
if "Date added" in df.columns:
    df["Date added"] = df["Date added"].fillna(date.today().isoformat())

# Split and write
ready = df[df["required_ok"]].copy()
needs = df[~df["required_ok"]].copy()
ready.to_csv(OUT_READY, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
needs.to_csv(OUT_NEEDS, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


print(f"✅ Wrote {OUT_READY} ({len(ready)} rows) and {OUT_NEEDS} ({len(needs)} rows))")
