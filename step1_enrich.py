#!/usr/bin/env python3
# step1_enrich.py
# Fills BPM and Mode using Deezer, AcousticBrainz, GetSongBPM, and iTunes+local analysis.

import os, io, re, json, time, math, requests, traceback, tempfile
import pandas as pd
import importlib
from dotenv import load_dotenv

# ---------- Missing-value helper ----------
def is_missing(x):
    """Treat None/''/NaN as missing."""
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

# ---------- Optional local analysis backends (dynamic) ----------
USE_MADMOM = False
USE_PYACA  = False

RNNBeatProcessor = None
TempoEstimationProcessor = None
computeKey = None
np = None
sf = None
librosa = None
AudioSegment = None

# madmom (tempo)
try:
    beats_mod = importlib.import_module("madmom.features.beats")     # type: ignore
    tempo_mod = importlib.import_module("madmom.features.tempo")     # type: ignore
    RNNBeatProcessor = getattr(beats_mod, "RNNBeatProcessor")
    TempoEstimationProcessor = getattr(tempo_mod, "TempoEstimationProcessor")
    USE_MADMOM = True
except Exception:
    pass

# pyACA (key→mode) + deps
try:
    np = importlib.import_module("numpy")                            # type: ignore
    sf = importlib.import_module("soundfile")                        # type: ignore
    try:
        computeKey = importlib.import_module("pyACA.computeKey").computeKey  # type: ignore
    except Exception:
        computeKey = importlib.import_module("pyACA").computeKey              # type: ignore
    USE_PYACA = True
except Exception:
    pass

# Optional fallbacks
try:
    librosa = importlib.import_module("librosa")                     # type: ignore
    AudioSegment = importlib.import_module("pydub").AudioSegment     # type: ignore
except Exception:
    pass

# Force pydub to use your ffmpeg if available
FFMPEG_PATH = r"C:\Users\ryanz\OneDrive\Desktop\ffmpeg\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
try:
    if AudioSegment and os.path.exists(FFMPEG_PATH):
        AudioSegment.converter = FFMPEG_PATH
except Exception:
    pass

# ---------- Config & env ----------
load_dotenv()
GETSONGBPM_API_KEY = os.getenv("GETSONGBPM_API_KEY", "").strip()

INPUT_CSV  = "step1_spotify_output.csv"
OUTPUT_CSV = "step1_enriched.csv"
CACHE_JSON = "step1_enrich_cache.json"

# polite default pacing
SLEEP_SHORT = 0.20

# ---------- Utilities ----------
def log(*args):
    print(*args, flush=True)

def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None so JSON stays valid."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj

def load_cache():
    if os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _sanitize_for_json(data)
        except Exception:
            pass
    return {}

def save_cache(cache):
    tmp = CACHE_JSON + ".tmp"
    safe = _sanitize_for_json(cache)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, CACHE_JSON)

def norm_key(title, artist):
    t = re.sub(r"\s*\(.*?\)|\s*-\s*(Remaster(ed)?|Live|Acoustic|Radio Edit|Mono|Stereo).*?$", "", title, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    a = re.sub(r"\s+", " ", artist).strip()
    return f"{t}|||{a}".lower()

def coerce_mode_from_key_str(keystr):
    # Accept strings like "C major", "A Minor", "G#m", "D minor"
    if not keystr:
        return None
    s = str(keystr).lower()
    if "major" in s or (s.endswith("m") is False and "#" in s is False and "minor" not in s):
        return "Major"
    if "minor" in s or s.endswith("m"):
        return "Minor"
    return None

def coerce_mode_from_scale(scale):
    # AcousticBrainz scale is "major" / "minor"
    if isinstance(scale, str):
        s = scale.lower().strip()
        if s == "major": return "Major"
        if s == "minor": return "Minor"
    if isinstance(scale, (int, float)):
        if int(scale) == 1: return "Major"
        if int(scale) == 0: return "Minor"
    return None

# ----- Helper: convert MP3 bytes -> temp WAV (for analyzers) -----
def write_temp_wav_from_mp3_bytes(mp3_bytes: bytes):
    """Convert MP3 bytes to a temporary WAV file path. Needs pydub + ffmpeg."""
    if not AudioSegment:
        return None
    tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        tmp_mp3.write(mp3_bytes); tmp_mp3.close()
        wav_path = tmp_mp3.name.replace(".mp3", ".wav")
        AudioSegment.from_file(tmp_mp3.name, format="mp3").export(wav_path, format="wav")
        try: os.remove(tmp_mp3.name)
        except Exception: pass
        return wav_path
    except Exception:
        try: os.remove(tmp_mp3.name)
        except Exception: pass
        return None

# ----- Local analyzers: BPM (madmom→librosa), Mode (pyACA→librosa) -----
def analyze_file_bpm_madmom(wav_path):
    if not (USE_MADMOM and wav_path):
        return None
    try:
        act   = RNNBeatProcessor()(wav_path)
        tempi = TempoEstimationProcessor()(act)  # [[bpm, weight], ...]
        if hasattr(tempi, "__len__") and len(tempi) > 0:
            return float(tempi[0][0])
    except Exception:
        return None
    return None

def analyze_file_bpm_librosa(wav_path):
    if not (librosa and wav_path):
        return None
    try:
        import numpy as np
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        if tempo is not None and len(tempo):
            return float(np.median(tempo))
    except Exception as e:
        print(f"⚠️ Librosa BPM error: {e}")
    return None


def analyze_file_mode_pyaca(wav_path):
    if not (USE_PYACA and sf and np and computeKey and wav_path):
        return None
    try:
        x, fs = sf.read(wav_path, always_2d=False)
        if x is None or fs is None:
            return None
        if hasattr(x, "ndim") and getattr(x, "ndim", 1) > 1:
            x = np.mean(x, axis=1)
        key_str, strength = computeKey(x, fs)  # e.g., "C major"
        return coerce_mode_from_key_str(key_str)
    except Exception:
        return None

def analyze_file_mode_librosa(wav_path):
    if not (librosa and wav_path):
        return None
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        maj  = chroma[[0,4,7], :].mean()   # C-E-G proxy
        minr = chroma[[0,3,7], :].mean()   # C-Eb-G proxy
        return "Major" if maj >= minr else "Minor"
    except Exception:
        return None

# ---------- Provider 1: Deezer (BPM) ----------
def fetch_deezer_bpm(title, artist, retries=2):
    # No key required. API: https://api.deezer.com/search?q=track:"..." artist:"..."
    q = f'track:"{title}" artist:"{artist}"'
    url = "https://api.deezer.com/search"
    params = {"q": q, "limit": 1}
    for attempt in range(retries+1):
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json() or {}
                items = data.get("data") or []
                if items:
                    it = items[0]
                    bpm = it.get("bpm")
                    # Deezer sometimes returns 0; treat as missing
                    if bpm and float(bpm) > 0:
                        return float(bpm)
                    return None
            elif r.status_code in (429, 500, 502, 503):
                time.sleep(0.6 * (attempt + 1))
                continue
        except Exception:
            time.sleep(0.4)
    return None

# ---------- Provider 2: MusicBrainz → AcousticBrainz (Mode / Key / BPM) ----------
def fetch_musicbrainz_recording_id(title, artist, retries=2):
    # Public JSON API (rate-limited). We do a simple recording search.
    url = "https://musicbrainz.org/ws/2/recording"
    params = {"query": f'"{title}" AND artist:"{artist}"', "fmt": "json", "limit": 1}
    headers = {"User-Agent": "MusicTherapyDataPipeline/1.0 (contact: you@example.com)"}
    for attempt in range(retries+1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                j = r.json() or {}
                recs = j.get("recordings") or []
                if recs:
                    return recs[0].get("id")
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5)
    return None

def fetch_acousticbrainz_highlevel(mbid, retries=1):
    # AcousticBrainz high-level endpoint (if available)
    # Note: The project has reduced availability; handle 404/5xx gracefully.
    url = f"https://acousticbrainz.org/{mbid}/high-level"
    for attempt in range(retries+1):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json() or {}
            elif r.status_code in (404, 410):
                return None
            else:
                time.sleep(0.6 * (attempt + 1))
        except Exception:
            time.sleep(0.5)
    return None

def interpret_acousticbrainz_mode(data):
    # Try to read scale/mode from common places
    if isinstance(data, dict):
        tonal = data.get("tonal") or {}
        key_scale = None
        if isinstance(tonal, dict):
            ks = tonal.get("key_scale")
            if isinstance(ks, dict):
                key_scale = ks.get("value")
            elif isinstance(ks, str):
                key_scale = ks
        m = coerce_mode_from_scale(key_scale)
        if m: return m
        chords_scale = None
        cs = tonal.get("chords_scale")
        if isinstance(cs, dict):
            chords_scale = cs.get("value")
        elif isinstance(cs, str):
            chords_scale = cs
        m = coerce_mode_from_scale(chords_scale)
        if m: return m
    return None

def fetch_mode_from_acousticbrainz(title, artist):
    mbid = fetch_musicbrainz_recording_id(title, artist)
    if not mbid:
        return None, None, None
    data = fetch_acousticbrainz_highlevel(mbid)
    if not data:
        return None, None, None
    # Try to extract bpm if present (not always available)
    bpm = None
    try:
        bpm = data.get("rhythm", {}).get("bpm", {}).get("value")
        if bpm is not None:
            bpm = float(bpm)
    except Exception:
        bpm = None
    mode = interpret_acousticbrainz_mode(data)
    # Try to reconstruct key string if present
    key_str = None
    try:
        key_name = data.get("tonal", {}).get("key_key", {}).get("value")
        scale = data.get("tonal", {}).get("key_scale", {}).get("value")
        if key_name and scale:
            key_str = f"{key_name} {scale}"
    except Exception:
        pass
    return mode, key_str, bpm

# ---------- Provider 3: GetSongBPM ----------
def fetch_getsongbpm(title, artist, retries=2):
    if not GETSONGBPM_API_KEY:
        return None, None
    url = "https://api.getsongbpm.com/search/"
    params = {"api_key": GETSONGBPM_API_KEY, "type": "both", "lookup": f"{title} {artist}"}
    for attempt in range(retries+1):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                j = r.json() or {}
                items = j.get("search") or j.get("result") or []
                if items:
                    it = items[0]
                    bpm = it.get("tempo") or it.get("bpm")
                    key = it.get("key") or it.get("song_key")
                    mode = coerce_mode_from_key_str(key)
                    try:
                        if bpm is not None:
                            bpm = float(bpm)
                    except Exception:
                        bpm = None
                    return bpm, mode
            elif r.status_code in (429, 500, 502, 503):
                time.sleep(0.6 * (attempt + 1))
                continue
        except Exception:
            time.sleep(0.4)
    return None, None

# ---------- Provider 4: iTunes preview + Local analysis ----------
def itunes_preview_url(title, artist):
    url = "https://itunes.apple.com/search"
    params = {"term": f"{title} {artist}", "media": "music", "entity": "song", "limit": 1, "country": "US"}

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200 and r.json().get("results"):
            return r.json()["results"][0].get("previewUrl")
    except Exception:
        pass
    return None
def analyze_preview(title, artist, retries=1):
    """Fetch iTunes preview and analyze BPM/mode locally with librosa or madmom."""
    url = itunes_preview_url(title, artist)
    if not url:
        print(f"⚠️ No iTunes preview found for {title} – {artist}")
        return None, None

    wav_path = None
    for attempt in range(retries + 1):
        try:
            print(f"🎧 Downloading preview for {title} – {artist} (attempt {attempt + 1})", flush=True)
            r = requests.get(url, timeout=10, stream=True)
            audio = r.content
            # ⬇️  THIS IS THE NEW SIZE CHECK
            if not audio or len(audio) < 5000:
                print(f"⚠️ iTunes preview too small ({len(audio) if audio else 0} B) for {title} – {artist}", flush=True)
                return None, None

            wav_path = write_temp_wav_from_mp3_bytes(audio)
            bpm  = analyze_file_bpm_madmom(wav_path) or analyze_file_bpm_librosa(wav_path)
            mode = analyze_file_mode_pyaca(wav_path) or analyze_file_mode_librosa(wav_path)

            if bpm or mode:
                print(f"✅ iTunes preview analysis: {title} – {artist} | BPM={bpm}, Mode={mode}", flush=True)
            else:
                print(f"⚠️ Analysis returned no data for {title} – {artist}", flush=True)
            return bpm, mode

        except Exception as e:
            print(f"⚠️ Error analyzing preview for {title} – {artist}: {e}", flush=True)
            time.sleep(0.5)

        finally:
            if wav_path and os.path.exists(wav_path):
                try: os.remove(wav_path)
                except Exception: pass

    print(f"⚠️ All preview attempts failed for {title} – {artist}")
    return None, None


    wav_path = None
    for attempt in range(retries + 1):
        try:
            print(f"🎧 Downloading preview for {title} – {artist} (attempt {attempt + 1})", flush=True)
            r = requests.get(url, timeout=10, stream=True)
            audio = r.content
            if len(audio) < 5000:
                print(f"⚠️ Preview too small ({len(audio)} bytes) for {title} – {artist}; skipping.", flush=True)
                return None, None

            wav_path = write_temp_wav_from_mp3_bytes(audio)
            ...

            ...


            # Run local analyzers
            bpm = analyze_file_bpm_madmom(wav_path) or analyze_file_bpm_librosa(wav_path)
            mode = analyze_file_mode_pyaca(wav_path) or analyze_file_mode_librosa(wav_path)

            if bpm or mode:
                print(f"✅ Preview analysis complete: {title} – {artist} | BPM={bpm}, Mode={mode}", flush=True)
            else:
                print(f"⚠️ Analysis returned no data for {title} – {artist}", flush=True)

            return bpm, mode

        except Exception as e:
            print(f"⚠️ Error analyzing preview for {title} – {artist}: {e}", flush=True)
            time.sleep(0.5)

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    print(f"⚠️ All preview attempts failed for {title} – {artist}")
    return None, None

def fetch_deezer_preview_url(title, artist, retries=2):
    """Return a 30s MP3 preview url from Deezer search (public)."""
    q = f'track:"{title}" artist:"{artist}"'
    url = "https://api.deezer.com/search"
    params = {"q": q, "limit": 1}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=8)
            if r.status_code == 200:
                data = r.json() or {}
                items = data.get("data") or []
                if items:
                    return items[0].get("preview")  # direct MP3 30s
            elif r.status_code in (429, 500, 502, 503):
                time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.4)
    return None

def analyze_preview_from_url(preview_url, title, artist):
    """Download MP3 preview from a url, convert to wav, run local analyzers."""
    if not preview_url:
        return None, None
    wav_path = None
    try:
        r = requests.get(preview_url, timeout=10, stream=True)
        audio = r.content
        if not audio or len(audio) < 5000:
            print(f"⚠️ Preview too small ({len(audio) if audio else 0} B) for {title} – {artist}", flush=True)
            return None, None
        wav_path = write_temp_wav_from_mp3_bytes(audio)
        bpm  = analyze_file_bpm_madmom(wav_path) or analyze_file_bpm_librosa(wav_path)
        mode = analyze_file_mode_pyaca(wav_path) or analyze_file_mode_librosa(wav_path)
        if bpm or mode:
            print(f"✅ Deezer preview analysis: {title} – {artist} | BPM={bpm}, Mode={mode}", flush=True)
        else:
            print(f"⚠️ Deezer preview analysis returned no data for {title} – {artist}", flush=True)
        return bpm, mode
    except Exception as e:
        print(f"⚠️ Deezer preview analyze error for {title} – {artist}: {e}", flush=True)
        return None, None
    finally:
        if wav_path and os.path.exists(wav_path):
            try: os.remove(wav_path)
            except Exception: pass


# ---------- Enrichment chain (ONLY overwrite if missing; cache clean values) ----------
def enrich_row(row, cache):
    """
    Enrich a single song row with BPM and Mode.
    Tries: cache → Deezer numeric BPM → Deezer preview (local analysis) → GetSongBPM (if key) → iTunes preview (local).
    """
    title = str(row.get("Title", "")).strip()
    artist = str(row.get("Artist", "")).strip()
    key = norm_key(title, artist)

    # normalize current values
    bpm  = None if is_missing(row.get("BPM"))  else row.get("BPM")
    mode = None if is_missing(row.get("Mode")) else row.get("Mode")
    bpm_src, mode_src = None, None

    # 0) cache
    if key in cache:
        cached = cache[key] or {}
        if is_missing(bpm)  and not is_missing(cached.get("BPM")):
            bpm,  bpm_src  = cached.get("BPM"),  "cache"
        if is_missing(mode) and not is_missing(cached.get("Mode")):
            mode, mode_src = cached.get("Mode"), "cache"
        if (bpm is not None) and (mode is not None):
            row["BPM"], row["Mode"] = float(bpm), str(mode)
            row["BPM Source"], row["Mode Source"] = bpm_src, mode_src
            return row

    # -------------------- TRY PROVIDERS --------------------
    print("➡️  PROVIDERS PATH ENTERED", flush=True)

    # 1) Deezer numeric BPM (rare, but trivial to try)
    if is_missing(bpm):
        print("… trying Deezer BPM", flush=True)
        bpm_try = fetch_deezer_bpm(title, artist)
        if not is_missing(bpm_try):
            bpm = float(bpm_try); bpm_src = bpm_src or "Deezer(bpm)"
            log(f"Deezer BPM: {title} - {artist} → {bpm}")

    # 2) Deezer preview (HIGH YIELD) → local librosa/madmom
    if is_missing(bpm) or is_missing(mode):
        print("… trying Deezer preview", flush=True)
        dz_prev = fetch_deezer_preview_url(title, artist)
        if dz_prev:
            pbpm, pmode = analyze_preview_from_url(dz_prev, title, artist)
            if is_missing(bpm)   and (pbpm is not None):
                bpm = float(pbpm);   bpm_src  = bpm_src  or ("madmom" if USE_MADMOM else "librosa(deezer)")
                log(f"Deezer Preview BPM: {title} - {artist} → {bpm}")
            if is_missing(mode) and (pmode is not None):
                mode = str(pmode); mode_src = mode_src or ("pyACA"  if USE_PYACA  else "librosa(deezer)")
                log(f"Deezer Preview Mode: {title} - {artist} → {mode}")
        else:
            print("   (no Deezer preview url)", flush=True)

    # 3) GetSongBPM (only if you set the key)
    if (is_missing(bpm) or is_missing(mode)) and GETSONGBPM_API_KEY:
        print("… trying GetSongBPM", flush=True)
        gbpm, gmode = fetch_getsongbpm(title, artist)
        if is_missing(bpm) and not is_missing(gbpm):
            bpm = float(gbpm); bpm_src = bpm_src or "GetSongBPM"
            log(f"GSBPM BPM: {title} - {artist} → {bpm}")
        if is_missing(mode) and not is_missing(gmode):
            mode = str(gmode); mode_src = mode_src or "GetSongBPM"
            log(f"GSBPM Mode: {title} - {artist} → {mode}")

    # 4) iTunes preview (sometimes blocked; still try)
    if is_missing(bpm) or is_missing(mode):
        print("… trying iTunes preview", flush=True)
        pbpm, pmode = analyze_preview(title, artist)
        if is_missing(bpm) and (pbpm is not None):
            bpm = float(pbpm); bpm_src = bpm_src or ("madmom" if USE_MADMOM else "librosa(itunes)")
            log(f"iTunes Preview BPM: {title} - {artist} → {bpm}")
        if is_missing(mode) and (pmode is not None):
            mode = str(pmode); mode_src = mode_src or ("pyACA"  if USE_PYACA  else "librosa(itunes)")
            log(f"iTunes Preview Mode: {title} - {artist} → {mode}")

    # -------------------- FINALIZE --------------------
    try:
        if isinstance(bpm, str):
            bpm = float(bpm.strip()) if bpm.strip() else None
    except Exception:
        bpm = None

    bpm_clean  = None if is_missing(bpm)  else float(bpm)
    mode_clean = None if is_missing(mode) else str(mode)

    row["BPM"] = bpm_clean
    row["Mode"] = mode_clean
    row["BPM Source"]  = bpm_src
    row["Mode Source"] = mode_src

    # cache clean values for reuse
    cache[key] = {"BPM": bpm_clean, "Mode": mode_clean}
    return row

# ---------- Main ----------
def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"❌ {INPUT_CSV} not found. Run your step1 search first.")
    df = pd.read_csv(INPUT_CSV)

    # Normalize NaN-like values to None
    df = df.replace({pd.NA: None})
    df = df.where(pd.notna(df), None)

    # tolerate either "Source link" or "Source Link"
    if "Source link" in df.columns and "Source Link" not in df.columns:
        df.rename(columns={"Source link": "Source Link"}, inplace=True)

    cache = load_cache()
    rows = []
    log("🚀 Enriching BPM/Mode via Deezer → AcousticBrainz → GetSongBPM → iTunes+local")
    for i, (_, r) in enumerate(df.iterrows(), 1):  # limit to 5 rows for debug
        title, artist = r.get("Title"), r.get("Artist")
        print(f"\n🎵 Processing {i}/{len(df)}: {title} – {artist}", flush=True)
        try:
            r = enrich_row(r, cache)
            print(f"✅ Finished {title} – {artist}", flush=True)
        except Exception as e:
            print(f"⚠️ Enrich error on {title} – {artist}: {e}", flush=True)
            print(traceback.format_exc())
        rows.append(r)
        time.sleep(SLEEP_SHORT)  # be nice to public APIs
        if i % 25 == 0:
            save_cache(cache)


    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    save_cache(cache)
    log(f"✅ Saved {OUTPUT_CSV} with {len(out)} rows.")
    missing_bpm = int(out['BPM'].isna().sum()) if 'BPM' in out.columns else 0
    missing_mode = int(out['Mode'].isna().sum()) if 'Mode' in out.columns else 0
    log(f"ℹ️ Missing BPM: {missing_bpm} | Missing Mode: {missing_mode}")

if __name__ == "__main__":
    main()
