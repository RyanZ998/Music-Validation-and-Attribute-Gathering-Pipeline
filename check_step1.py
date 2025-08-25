import pandas as pd

# Step 1 outputs
try:
    df = pd.read_csv("step1_spotify_output.csv")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))
    print(df.head(3))
except FileNotFoundError:
    print("No step1_spotify_output.csv found.")

try:
    hits = pd.read_csv("step1_search_hits.csv")
    print("\nSearch hits columns:", list(hits.columns))
    print("Search hits rows:", len(hits))
except Exception as e:
    print("\nNo step1_search_hits.csv or error:", e)

try:
    with open("step1_audio_features_failed.txt","r",encoding="utf-8") as f:
        bad = f.read().strip().splitlines()
        print("\nAudio features failed IDs:", len(bad))
except FileNotFoundError:
    print("\nNo step1_audio_features_failed.txt")
