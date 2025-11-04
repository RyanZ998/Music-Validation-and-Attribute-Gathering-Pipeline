#!/usr/bin/env python3
import os, re, csv, json, time, random
import pandas as pd
from dotenv import load_dotenv

# ---------- Config ----------
INPUT_CSV  = "step1_enriched.csv"
OUTPUT_CSV = "step2_lyrics_output.csv"
CACHE_JSON = "lyrics_cache.json"       # caches lyrics so we don't re-hit Genius
NRC_VAD_PATH = "NRC-VAD-Lexicon.txt"   # optional (if missing, we fall back)

REQUEST_SLEEP = (0.6, 1.2)  # polite randomized pacing between Genius calls

# ---------- Optional analyzers ----------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except Exception:
    _VADER = None
    _HAS_VADER = False

try:
    from textblob import TextBlob
    _HAS_TB = True
except Exception:
    _HAS_TB = False

# ---------- Genius client (optional) ----------
load_dotenv()
try:
    import lyricsgenius
    _HAS_GENIUS = True
except Exception:
    _HAS_GENIUS = False

GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")
genius = None
if _HAS_GENIUS and GENIUS_API_KEY:
    genius = lyricsgenius.Genius(GENIUS_API_KEY, timeout=15, retries=3)
    genius.remove_section_headers = True
    genius.skip_non_songs = True
    genius.excluded_terms = ["(Remix)", "(Live)"]
else:
    print("⚠️ Genius disabled (missing package or API key). Will skip lyrics fetch.", flush=True)

# ---------- Helpers ----------
_word_re = re.compile(r"[A-Za-z']+")
def log(*a): print(*a, flush=True)

def tokenize(text: str):
    return _word_re.findall(text.lower()) if isinstance(text, str) else []

def clean_lyrics(txt: str) -> str:
    if not isinstance(txt, str): return ""
    txt = re.sub(r"(?mi)^.*(embed|you might also like|contributors).*$", "", txt)
    txt = re.sub(r"\[.*?\]", " ", txt)     # remove [Chorus], [Verse], etc.
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# ---- NRC-VAD loader (valence/arousal on 0..1 scale) ----
_VAD = {}
def load_vad():
    global _VAD
    if not os.path.exists(NRC_VAD_PATH):
        return
    with open(NRC_VAD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"\s+", s)  # tabs or spaces
            if len(parts) < 3:
                continue
            try:
                w = parts[0].lower()
                v = float(parts[1]); a = float(parts[2])  # already 0..1
                _VAD[w] = (v, a)
            except Exception:
                continue
load_vad()
_HAS_VAD = len(_VAD) > 0

def vad_from_tokens(tokens):
    if not _HAS_VAD or not tokens:
        return (None, None)
    vals, aros = [], []
    for t in tokens:
        va = _VAD.get(t)
        if va:
            vals.append(va[0]); aros.append(va[1])
    if not vals:
        return (None, None)
    return (sum(vals)/len(vals), sum(aros)/len(aros))

def vader_valence_01(text):
    if not (_HAS_VADER and isinstance(text, str) and text.strip()):
        return None
    try:
        c = _VADER.polarity_scores(text)["compound"]  # -1..1
        return (c + 1.0) / 2.0  # → 0..1
    except Exception:
        return None

def textblob_valence_01(text):
    if not (_HAS_TB and isinstance(text, str) and text.strip()):
        return None
    try:
        pol = TextBlob(text).sentiment.polarity  # -1..1
        return (pol + 1.0) / 2.0
    except Exception:
        return None

# ---------- Cache ----------
def load_cache():
    if os.path.exists(CACHE_JSON):
        try:
            return json.load(open(CACHE_JSON, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache):
    tmp = CACHE_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CACHE_JSON)

def ck(title, artist):
    return f"{title.strip().lower()}|||{artist.strip().lower()}"

def get_lyrics(title: str, artist: str, cache: dict) -> str:
    key = ck(title, artist)
    cached = cache.get(key) or {}
    if "lyrics" in cached and cached["lyrics"]:
        return cached["lyrics"]

    text = ""
    if genius:
        try:
            song = genius.search_song(title=title, artist=artist)
            if song and song.lyrics:
                text = clean_lyrics(song.lyrics)
        except Exception as e:
            log(f"⚠️ Genius error for {title} – {artist}: {e}")
            text = ""

    cache[key] = {"lyrics": text, **cached}
    time.sleep(random.uniform(*REQUEST_SLEEP))  # polite
    return text

# ---------- Main ----------
def _to_float_or_none(x):
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"❌ {INPUT_CSV} not found. Run enrich first.")

    df = pd.read_csv(INPUT_CSV)
    for c in ("Title","Artist"):
        if c not in df.columns:
            raise SystemExit("❌ step1_enriched.csv must have Title and Artist")

    cache = load_cache()

    lyrics_col = []
    val_col, aro_col = [], []

    log("🚀 Step 2: fetching lyrics & computing valence/arousal …")
    for i, row in df.iterrows():
        title = str(row["Title"])
        artist = str(row["Artist"])
        key = ck(title, artist)

        # 1) get lyrics
        text = get_lyrics(title, artist, cache)

        # 2) compute sentiment — but only trust NUMERIC cached values
        cached = cache.get(key) or {}
        v_cached = _to_float_or_none(cached.get("valence"))
        a_cached = _to_float_or_none(cached.get("arousal"))

        if v_cached is None or a_cached is None:
            tokens = tokenize(text)
            v_nrc, a_nrc = vad_from_tokens(tokens)
            v = v_nrc if v_nrc is not None else vader_valence_01(text) or textblob_valence_01(text)
            a = a_nrc  # if NRC missing, leave None (that’s okay)
            # save back to cache as NUMBERS only
            cache[key] = {"lyrics": text, "valence": _to_float_or_none(v), "arousal": _to_float_or_none(a)}
        else:
            v, a = v_cached, a_cached
            # ensure lyrics also present in cache
            if "lyrics" not in cached or not cached["lyrics"]:
                cache[key]["lyrics"] = text

        lyrics_col.append(text)
        val_col.append(_to_float_or_none(v))
        aro_col.append(_to_float_or_none(a))

        if (i+1) % 25 == 0:
            save_cache(cache)
            log(f"… processed {i+1}/{len(df)}")

    # 3) write out exactly what the rubric expects
    df["Lyrics"] = lyrics_col
    df["Lyric sentiment valence"] = pd.to_numeric(val_col, errors="coerce")
    df["Lyric sentiment arousal"] = pd.to_numeric(aro_col, errors="coerce")

    df.to_csv(OUTPUT_CSV, index=False)
    save_cache(cache)
    log(f"✅ Step 2 done → {OUTPUT_CSV}")
