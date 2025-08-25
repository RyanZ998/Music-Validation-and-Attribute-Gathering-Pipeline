# step3_gpt_fill.py
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise RuntimeError("Missing OPENAI_API_KEY in .env file")

client = OpenAI(api_key=openai_key)

df = pd.read_csv("step2_lyrics_output.csv")

contexts, contraindications = [], []

for _, row in df.iterrows():
    prompt = f"""
    Song: {row['Title']} by {row['Artist']}
    BPM: {row['BPM']}, Mode: {row['Mode']}, Valence: {row['Valence (Spotify)']}
    Lyrics sentiment: valence={row['Lyric sentiment valence']}, arousal={row['Lyric sentiment arousal']}
    
    Based on depression therapy guidelines:
    1. Suggest an appropriate listening context.
    2. Identify potential contraindications.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    parts = text.split("\n")
    contexts.append(parts[0].replace("1.", "").strip() if parts else "")
    contraindications.append(parts[1].replace("2.", "").strip() if len(parts) > 1 else "")

df["Suggested listening context"] = contexts
df["Contraindications"] = contraindications

df.to_csv("step3_full_dataset.csv", index=False)
print("✅ Step 3 done → step3_full_dataset.csv")
