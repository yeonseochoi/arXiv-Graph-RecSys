import os
import pandas as pd

INPUT_PATH = "two_tower/data/rated_papers_with_embedding_idx.csv"
OUTPUT_DIR = "two_tower/data"

MIN_USER_INTERACTIONS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "user_id": str,
        "arxiv_id": str,
        "paper_id": str,
    },
    low_memory=False,
)

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])

# Keep users with enough history
user_counts = df["user_id"].value_counts()
valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
df = df[df["user_id"].isin(valid_users)].copy()

df = df.sort_values(["user_id", "time"])

train_parts = []
valid_parts = []
test_parts = []

for user_id, g in df.groupby("user_id"):
    g = g.sort_values("time")
    n = len(g)

    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    # Safety: ensure valid/test not empty for users with 5+ interactions
    train_parts.append(g.iloc[:train_end])
    valid_parts.append(g.iloc[train_end:valid_end])
    test_parts.append(g.iloc[valid_end:])

train_df = pd.concat(train_parts).reset_index(drop=True)
valid_df = pd.concat(valid_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
valid_df.to_csv(os.path.join(OUTPUT_DIR, "valid.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("After filtering:")
print("users:", df["user_id"].nunique())
print("interactions:", len(df))
print()

print("Train:", len(train_df), train_df["user_id"].nunique())
print("Valid:", len(valid_df), valid_df["user_id"].nunique())
print("Test :", len(test_df), test_df["user_id"].nunique())

print()
print("Label distribution:")
print("train")
print(train_df["label"].value_counts(normalize=True))
print("valid")
print(valid_df["label"].value_counts(normalize=True))
print("test")
print(test_df["label"].value_counts(normalize=True))