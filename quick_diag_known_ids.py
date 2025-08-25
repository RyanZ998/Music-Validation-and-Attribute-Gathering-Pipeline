# quick_diag_known_ids.py
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

ids = [
    "3AJwUDP919kvQ9QcozQPxg",  # Coldplay - Fix You (album)
    "6habFhsOp2NvshLv26DqMb",  # Pharrell - Happy
]

for tid in ids:
    try:
        f = sp.audio_features([tid])[0]
        print("features:", tid, bool(f), f and (f.get("tempo"), f.get("mode"), f.get("valence")))
    except Exception as e:
        print("features error:", e)

    try:
        a = sp.audio_analysis(tid)
        t = a.get("track", {})
        print("analysis:", tid, (t.get("tempo"), t.get("mode")))
    except Exception as e:
        print("analysis error:", e)
