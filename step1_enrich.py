#!/usr/bin/env python3
# step1_enrich.py
# Fills BPM and Mode using Deezer, AcousticBrainz, GetSongBPM, and iTunes+local analysis.

import os, io, re, json, time, math, requests, traceback
import pandas as pd

# ---------- Optional local analysis backends ----------
# Prefer Essentia if available; otherwise fall back to librosa (tempo) + crude mode
USE_ESSENTIA = False
try:
    import essentia.standard as es
    USE_ESSENTIA = True
except Exception:
    try:
        import librosa
        from pydub import AudioSegment
    except Exception:
        librosa = None
        AudioSegment = None

# ---------- Config & env ----------
from dotenv import load_dotenv
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

def load_cache():
    if os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache(cache):
    tmp = CACHE_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
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
        # some datasets encode 1=major 0=minor
        if int(scale) == 1: return "Major"
        if int(scale) == 0: return "Minor"
    return None

# ---------- Provider 1: Deezer (BPM) ----------
def fetch_deezer_bpm(title, artist, retries=2):
    # No key required. API: https://api.deezer.com/search?q=track:"..." artist:"..."
    q = f'track:"{title}" artist:"{artist}"'
    url = "https://api.deezer.com/search"
    params = {"q": q, "limit": 1}
    for attempt in range(retries+1):
        try:
            r = requests.get(url, params=params, timeout=15)
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
    # Structure varies across dumps; try a few keys safely.
    candidates = []
    # Key estimator (e.g., "tonal.key_key": "C", "tonal.key_scale": "minor")
    if isinstance(data, dict):
        tonal = data.get("tonal") or {}
        key_scale = None
        if isinstance(tonal, dict):
            # some formats: tonal.key_scale -> {"value": "minor"}
            ks = tonal.get("key_scale")
            if isinstance(ks, dict):
                key_scale = ks.get("value")
            elif isinstance(ks, str):
                key_scale = ks
        m = coerce_mode_from_scale(key_scale)
        if m: return m
        # as a fallback, parse "chords_key" + "chords_scale"
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
                    # They sometimes return 'tempo' or 'bpm'
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
    params = {"term": f"{title} {artist}", "media": "music", "entity": "song", "limit": 1}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200 and r.json().get("results"):
            return r.json()["results"][0].get("previewUrl")
    except Exception:
        pass
    return None

def analyze_from_bytes_with_essentia(mp3_bytes):
    loader = es.EasyLoader(filename=None)  # we’ll use MonoLoader below
    # Write bytes to temp wav via essentia workflow (essentia prefers files; use temp)
    # Simpler: use MonoLoader on bytes via temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        mp3_path = f.name
    try:
        mono = es.MonoLoader(filename=mp3_path, sampleRate=44100)()
        rhythm = es.RhythmExtractor2013()(mono)
        tempo = float(rhythm[0])  # bpm
        # Mode via KeyExtractor
        key, scale, strength = es.KeyExtractor()(mono)
        mode = coerce_mode_from_scale(scale)
        return tempo, mode
    finally:
        try:
            os.remove(mp3_path)
        except Exception:
            pass

def analyze_from_bytes_with_librosa(mp3_bytes):
    if not (librosa and AudioSegment):
        return None, None
    from io import BytesIO
    seg = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
    wav = BytesIO()
    seg.export(wav, format="wav")
    wav.seek(0)
    y, sr = librosa.load(wav, sr=None, mono=True)
    # Tempo: average across frames
    tempos = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
    tempo = float(tempos.mean()) if tempos is not None and len(tempos) else None
    # Crude mode: compare triadic energy near C major vs C minor—very approximate
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    maj = chroma[[0,4,7], :].mean()    # C-E-G bins
    minr = chroma[[0,3,7], :].mean()   # C-Eb-G bins
    mode = "Major" if maj >= minr else "Minor"
    if tempo is not None:
        tempo = round(tempo, 1)
    return tempo, mode

def analyze_preview(title, artist, retries=1):
    url = itunes_preview_url(title, artist)
    if not url:
        return None, None
    for attempt in range(retries+1):
        try:
            audio = requests.get(url, timeout=20).content
            if USE_ESSENTIA:
                return analyze_from_bytes_with_essentia(audio)
            else:
                return analyze_from_bytes_with_librosa(audio)
        except Exception:
            time.sleep(0.5)
    return None, None

# ---------- Enrichment chain ----------
def enrich_row(row, cache):
    title = str(row.get("Title", "")).strip()
    artist = str(row.get("Artist", "")).strip()
    key = norm_key(title, artist)

    # if already cached, use it
    if key in cache:
        cached = cache[key]
        row["BPM"]  = cached.get("BPM")  if row.get("BPM") in (None, "", float("nan")) else row.get("BPM")
        row["Mode"] = cached.get("Mode") if not row.get("Mode") else row.get("Mode")
        return row

    bpm = row.get("BPM")
    mode = row.get("Mode")

    # 1) Deezer BPM
    if not bpm:
        bpm = fetch_deezer_bpm(title, artist)
        if bpm: log(f"Deezer BPM: {title} - {artist} → {bpm}")

    # 2) AcousticBrainz mode/key/bpm
    if not mode or not bpm:
        ab_mode, ab_keystr, ab_bpm = fetch_mode_from_acousticbrainz(title, artist)
        if ab_mode and not mode:
            mode = ab_mode
            log(f"AB Mode: {title} - {artist} → {mode}")
        if ab_bpm and not bpm:
            bpm = ab_bpm
            log(f"AB BPM: {title} - {artist} → {bpm}")
        if (not mode) and ab_keystr:
            m = coerce_mode_from_key_str(ab_keystr)
            if m:
                mode = m
                log(f"AB Key→Mode: {title} - {artist} → {mode}")

    # 3) GetSongBPM
    if (not bpm) or (not mode):
        gbpm, gmode = fetch_getsongbpm(title, artist)
        if gbpm and not bpm:
            bpm = gbpm
            log(f"GSBPM BPM: {title} - {artist} → {bpm}")
        if gmode and not mode:
            mode = gmode
            log(f"GSBPM Mode: {title} - {artist} → {mode}")

    # 4) iTunes preview + local analysis
    if (bpm is None) or (mode is None):
        pbpm, pmode = analyze_preview(title, artist)
        if pbpm and not bpm:
            bpm = pbpm
            log(f"Preview BPM: {title} - {artist} → {bpm}")
        if pmode and not mode:
            mode = pmode
            log(f"Preview Mode: {title} - {artist} → {mode}")

    # finalize
    if isinstance(bpm, str):
        try: bpm = float(bpm)
        except Exception: bpm = None
    row["BPM"] = bpm
    row["Mode"] = mode

    # cache
    cache[key] = {"BPM": bpm, "Mode": mode}
    return row

def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"❌ {INPUT_CSV} not found. Run your step1 search first.")
    df = pd.read_csv(INPUT_CSV)

    # tolerate either "Source link" or "Source Link"
    if "Source link" in df.columns and "Source Link" not in df.columns:
        df.rename(columns={"Source link": "Source Link"}, inplace=True)

    cache = load_cache()
    rows = []
    log("🚀 Enriching BPM/Mode via Deezer → AcousticBrainz → GetSongBPM → iTunes+local")
    for i, (_, r) in enumerate(df.iterrows(), 1):
        try:
            r = enrich_row(r, cache)
        except Exception as e:
            log(f"⚠️ Enrich error on row {i}: {r.get('Title')} - {r.get('Artist')}")
            log(traceback.format_exc())
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
