#!/usr/bin/env python3
# step3_gpt_fill.py
# Fills "Suggested listening context" and "Contraindications" using a conservative, JSON-only GPT prompt.
# - Strict JSON parsing (no brittle newline splits)
# - Caching to avoid repeat costs
# - Backoff/retries for rate limits
# - Safety: non-clinical, research framing

import os, json, time, math
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# -------------------- Setup --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

# pip install openai==1.*  (the new SDK)
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_CSV  = "step2_lyrics_output.csv"   # must include Title, Artist, BPM, Mode, Lyric sentiment valence, Lyric sentiment arousal (names can be tweaked below)
OUTPUT_CSV = "step3_full_dataset.csv"
CACHE_JSON = Path("step3_gpt_cache.json")

MODEL = "gpt-4o-mini"  # change if you prefer another model
N_RETRIES = 4
BASE_SLEEP = 1.0

# -------------------- Column mapping --------------------
# If your CSV has slightly different headers, adjust here:
COL_TITLE   = "Title"
COL_ARTIST  = "Artist"
COL_BPM     = "BPM"
COL_MODE    = "Mode"
COL_S_VAL   = "Valence (Spotify)"              # optional; some rows may be blank
COL_L_VAL   = "Lyric sentiment valence"        # from step 2
COL_L_ARO   = "Lyric sentiment arousal"        # from step 2
COL_OUT_CTX = "Suggested listening context"
COL_OUT_CON = "Contraindications"

# -------------------- Prompting --------------------
SYSTEM = (
    "You are assisting a music-therapy research workflow for non-clinical use. "
    "Your job: Given a song's metadata and lyric affect (valence/arousal), "
    "propose a conservative listening context and clear contraindications. "
    "Do not give medical advice. Do not diagnose. Avoid prescriptive claims. "
    "Prefer low-risk, opt-in contexts (e.g., solo listening with breathing guidance). "
    "If risk cues exist (e.g., violent/explicit/self-harm themes), flag them as contraindications. "
    "Return strictly valid JSON with keys: "
    "listening_context (string), contraindications (string), rationale (string <= 140 chars)."
)

USER_TMPL = (
    "Song: {title} — {artist}\n"
    "Attributes:\n"
    "- BPM: {bpm}\n"
    "- Mode: {mode}\n"
    "- Spotify valence: {sval}\n"
    "- Lyric valence: {lval}\n"
    "- Lyric arousal: {laro}\n\n"
    "Task:\n"
    "1) Suggest one concise listening_context string suitable for a cautious, non-clinical setting for depression-focused research (e.g., 'eyes-closed, paced breathing, dim light, 10 min').\n"
    "2) List contraindications as a concise string (e.g., 'avoid during acute panic; explicit content may trigger rumination').\n"
    "3) Provide a short rationale (<=140 chars).\n\n"
    "Constraints:\n"
    "- No medical advice. No guarantees of efficacy. Conservative tone.\n"
    "- Prefer downshifting arousal if lval<0 or laro>0.7.\n"
    "- Output strict JSON only."
)

def sanitize_float(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        return float(x)
    except Exception:
        return None

def load_cache():
    if CACHE_JSON.exists():
        try:
            return json.loads(CACHE_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache):
    CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def call_model(payload):
    # Backoff + retries for rate limits/network blips
    sleep = BASE_SLEEP
    for attempt in range(N_RETRIES):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role":"system","content": SYSTEM},
                    {"role":"user","content": payload}
                ],
            )
            content = r.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            msg = str(e)
            # Basic rate-limit/backoff handling
            if "RateLimit" in msg or "429" in msg or "insufficient_quota" in msg:
                # If quota truly exhausted, no point retrying too long
                if "insufficient_quota" in msg:
                    raise RuntimeError(
                        "OpenAI quota exhausted. Add billing or reduce usage."
                    )
                time.sleep(sleep)
                sleep *= 2
                continue
            # transient network/server error
            if attempt < N_RETRIES - 1:
                time.sleep(sleep)
                sleep *= 2
                continue
            raise
    return None

def main():
    # Load data
    df = pd.read_csv(INPUT_CSV)
    # Normalize NaNs to None
    df = df.where(pd.notna(df), None)

    # Prepare outputs (preserve existing columns if re-running)
    if COL_OUT_CTX not in df.columns:
        df[COL_OUT_CTX] = None
    if COL_OUT_CON not in df.columns:
        df[COL_OUT_CON] = None

    cache = load_cache()
    updates = 0

    for idx, row in df.iterrows():
        title  = (row.get(COL_TITLE)  or "").strip()
        artist = (row.get(COL_ARTIST) or "").strip()

        key = f"{title}|||{artist}"
        if not title or not artist:
            continue  # skip incomplete rows

        # Skip if already filled (idempotent)
        if row.get(COL_OUT_CTX) and row.get(COL_OUT_CON):
            continue

        # Build the user payload
        bpm  = sanitize_float(row.get(COL_BPM))
        mode = (row.get(COL_MODE) or "").strip() if row.get(COL_MODE) else None
        sval = sanitize_float(row.get(COL_S_VAL))
        lval = sanitize_float(row.get(COL_L_VAL))
        laro = sanitize_float(row.get(COL_L_ARO))

        user_msg = USER_TMPL.format(
            title=title, artist=artist,
            bpm=bpm if bpm is not None else "NA",
            mode=mode if mode else "NA",
            sval=f"{sval:.2f}" if sval is not None else "NA",
            lval=f"{lval:.2f}" if lval is not None else "NA",
            laro=f"{laro:.2f}" if laro is not None else "NA",
        )

        # Cache hit?
        if key in cache:
            ans = cache[key]
        else:
            ans = call_model(user_msg)
            if ans is None:
                continue
            # keep cache small/clean
            cache[key] = {
                "listening_context": ans.get("listening_context"),
                "contraindications": ans.get("contraindications"),
                "rationale": ans.get("rationale"),
            }
            if (idx + 1) % 10 == 0:
                save_cache(cache)

        # Write back to dataframe
        df.at[idx, COL_OUT_CTX] = ans.get("listening_context")
        df.at[idx, COL_OUT_CON] = ans.get("contraindications")
        updates += 1

    # Save outputs
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    save_cache(cache)
    print(f"✅ Step 3 done → {OUTPUT_CSV} (updated {updates} rows)")

if __name__ == "__main__":
    main()
