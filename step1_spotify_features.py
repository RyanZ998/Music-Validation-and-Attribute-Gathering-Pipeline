import os, sys, time
import pandas as pd
import spotipy
import time
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# ---------- Setup ----------
load_dotenv()
CID = os.getenv("SPOTIFY_CLIENT_ID")
CSECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
if not CID or not CSECRET:
    sys.exit("❌ Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in .env")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CID, client_secret=CSECRET))

# ---------- Load input ----------
try:
    df_in = pd.read_csv("songs.csv")
except FileNotFoundError:
    sys.exit("❌ songs.csv not found in this folder.")
if not {"Title","Artist"}.issubset(df_in.columns):
    sys.exit("❌ songs.csv must have columns: Title,Artist")

# Clean inputs
df_in["Title"] = df_in["Title"].astype(str).str.strip()
df_in["Artist"] = df_in["Artist"].astype(str).str.strip()
df_in = df_in[(df_in["Title"]!="") & (df_in["Artist"]!="")]

# ---------- Search ----------
print("🔎 Searching Spotify for tracks …")
hits_rows, not_found = [], []
for _, r in df_in.iterrows():
    title, artist = r["Title"], r["Artist"]
    query = f'track:"{title}" artist:"{artist}"'  # exact-ish search
    try:
        # You can remove market="US" if needed:
        res = sp.search(q=query, type="track", limit=1)  # , market="US"
        items = res.get("tracks", {}).get("items", [])
    except SpotifyException as e:
        print("⚠️ Search error:", title, "-", artist, "→", e)
        items = []

    if items:
        t = items[0]
        hits_rows.append({
            "Title": title,
            "Artist": artist,
            "Track ID": t["id"],
            "Source link": t["external_urls"]["spotify"],
            "Found Title": t["name"],
            "Found Artist(s)": ", ".join(a["name"] for a in t["artists"])
        })
    else:
        not_found.append(f"{title} - {artist}")

df_hits = pd.DataFrame(hits_rows)
if df_hits.empty:
    sys.exit("❌ No tracks found from your CSV. Check Title/Artist formatting.")
df_hits.to_csv("step1_search_hits.csv", index=False)

if not_found:
    with open("step1_not_found.txt","w",encoding="utf-8") as f:
        f.write("\n".join(not_found))
    print(f"⚠️ Not found: {len(not_found)} (see step1_not_found.txt)")

# ---------- Audio features (with fallback to audio_analysis) ----------
print("🎛  Fetching audio features (single-ID mode, with fallback) …")
features_rows, failed_ids = [], []

def get_features_single(track_id, retries=2, delay=0.5):
    """
    Try audio_features first; if it fails or returns None, fall back to audio_analysis.
    Returns a dict with keys: BPM, Mode, Valence (Spotify) — or None on total failure.
    """
    # 1) Primary: audio_features
    for attempt in range(retries + 1):
        try:
            f = sp.audio_features([track_id])[0]
            if f:
                return {
                    "BPM": f.get("tempo"),
                    "Mode": "Major" if f.get("mode") == 1 else "Minor" if f.get("mode") == 0 else None,
                    "Valence (Spotify)": f.get("valence"),
                }
            # If None, retry (rare transient) or fall through to fallback
        except SpotifyException as e:
            # e.g., 403 in some environments — don't fail hard; try again then fallback
            if attempt < retries:
                time.sleep(delay)
                continue
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
                continue
        break  # move to fallback

    # 2) Fallback: audio_analysis (tempo/mode/key live under analysis["track"])
    for attempt in range(retries + 1):
        try:
            a = sp.audio_analysis(track_id)
            t = a.get("track", {}) if isinstance(a, dict) else {}
            tempo = t.get("tempo")
            mode_val = t.get("mode")  # 1 = major, 0 = minor
            return {
                "BPM": tempo,
                "Mode": "Major" if mode_val == 1 else "Minor" if mode_val == 0 else None,
                # audio_analysis doesn't include valence — leave None
                "Valence (Spotify)": None,
            }
        except SpotifyException as e:
            if attempt < retries:
                time.sleep(delay)
                continue
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
                continue
            return None

for _, meta in df_hits.iterrows():
    tid = meta["Track ID"]
    f = get_features_single(tid)

    # Build output row, ensuring BPM/Mode/Valence columns exist
    row = {
        "Track ID": tid,
        "Title": meta["Title"],
        "Artist": meta["Artist"],
        "Source link": meta["Source link"],
        "BPM": None,
        "Mode": None,
        "Valence (Spotify)": None,
    }
    if f and isinstance(f, dict):
        row.update(f)
    else:
        failed_ids.append(tid)

    features_rows.append(row)
    # polite pacing
    time.sleep(0.1)

df_out = pd.DataFrame(features_rows)
# Make sure key columns exist even if all None (edge cases)
for col in ["BPM","Mode","Valence (Spotify)"]:
    if col not in df_out.columns:
        df_out[col] = None

df_out.to_csv("step1_spotify_output.csv", index=False)
print("✅ Saved step1_spotify_output.csv with", len(df_out), "rows")

if failed_ids:
    with open("step1_audio_features_failed.txt","w",encoding="utf-8") as f:
        f.write("\n".join(failed_ids))
    print(f"⚠️ {len(failed_ids)} IDs returned no audio features → step1_audio_features_failed.txt")
else:
    print("🎉 All IDs returned audio features.")
