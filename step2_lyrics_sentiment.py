# step2_lyrics_sentiment.py
import os
import pandas as pd
from textblob import TextBlob
import lyricsgenius
from dotenv import load_dotenv

load_dotenv()
genius_api_key = os.getenv("GENIUS_API_KEY")

if not genius_api_key:
    raise RuntimeError("Missing GENIUS_API_KEY in .env file")

genius = lyricsgenius.Genius(genius_api_key)

df = pd.read_csv("step1_spotify_output.csv")

lyrics, sentiments = [], []
for _, row in df.iterrows():
    try:
        song = genius.search_song(row["Title"], row["Artist"])
        if song:
            lyrics.append(song.lyrics)
            blob = TextBlob(song.lyrics)
            sentiments.append({
                "Lyric sentiment valence": blob.sentiment.polarity,
                "Lyric sentiment arousal": blob.sentiment.subjectivity
            })
        else:
            lyrics.append("")
            sentiments.append({"Lyric sentiment valence": None, "Lyric sentiment arousal": None})
    except Exception as e:
        print("Error:", e)
        lyrics.append("")
        sentiments.append({"Lyric sentiment valence": None, "Lyric sentiment arousal": None})

df["Lyrics"] = lyrics
df["Lyric sentiment valence"] = [s["Lyric sentiment valence"] for s in sentiments]
df["Lyric sentiment arousal"] = [s["Lyric sentiment arousal"] for s in sentiments]

df.to_csv("step2_lyrics_output.csv", index=False)
print("✅ Step 2 done → step2_lyrics_output.csv")
