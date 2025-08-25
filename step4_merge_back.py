import pandas as pd
from datetime import date

# Load original and final outputs
orig = pd.read_csv("songs.csv")  # must have Title, Artist
final = pd.read_csv("step3_full_dataset.csv")

# Normalize for safer matching
for df in (orig, final):
    df["Title_norm"] = df["Title"].str.strip().str.lower()
    df["Artist_norm"] = df["Artist"].str.strip().str.lower()

# Preserve original order
orig["_row"] = range(len(orig))

# Merge new fields back onto the original list
merged = orig.merge(
    final.drop_duplicates(subset=["Title_norm", "Artist_norm"]),
    on=["Title_norm", "Artist_norm"],
    how="left",
    suffixes=("", "_new")
)

# Drop helper cols
merged = merged.drop(columns=["Title_norm", "Artist_norm"])

# Optional: fill required minimal fields if missing
if "Curator" not in merged.columns:
    merged["Curator"] = "You"
if "Date added" not in merged.columns:
    merged["Date added"] = date.today().isoformat()

# Sort back to original order and save
merged = merged.sort_values("_row").drop(columns=["_row"])
merged.to_csv("FINAL_merged_dataset.csv", index=False)
print("✅ Created FINAL_merged_dataset.csv")
