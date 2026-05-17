import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================

EMB_PATH = "paper_embeddings.npy"

DATA_DIR = "two_tower/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train_arxividx.csv")
VALID_PATH = os.path.join(DATA_DIR, "valid_arxividx.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_arxividx.csv")

SAVE_DIR = "two_tower/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
USER_ID_DIM = 64
HIDDEN_DIM = 256
OUT_DIM = 128

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# User profiles
# =========================

def build_user_profiles(train_df, paper_embeddings, emb_dim):
    """
    Build user profile only from train interactions.
    liked_profile: mean embedding of liked papers
    disliked_profile: mean embedding of disliked papers
    """
    profiles = {}

    for user_id, g in train_df.groupby("user_id"):
        liked_idx = g.loc[g["label"] == 1, "row_idx"].astype(int).values
        disliked_idx = g.loc[g["label"] == 0, "row_idx"].astype(int).values

        if len(liked_idx) > 0:
            liked_vec = paper_embeddings[liked_idx].mean(axis=0)
        else:
            liked_vec = np.zeros(emb_dim, dtype=np.float32)

        if len(disliked_idx) > 0:
            disliked_vec = paper_embeddings[disliked_idx].mean(axis=0)
        else:
            disliked_vec = np.zeros(emb_dim, dtype=np.float32)

        profiles[user_id] = {
            "liked": liked_vec.astype(np.float32),
            "disliked": disliked_vec.astype(np.float32),
        }

    return profiles

# =========================
# Dataset
# =========================

class PaperRatingDataset(Dataset):
    def __init__(self, df, paper_embeddings, user_to_idx, user_profiles, emb_dim):
        self.df = df.reset_index(drop=True)
        self.paper_embeddings = paper_embeddings
        self.user_to_idx = user_to_idx
        self.user_profiles = user_profiles
        self.emb_dim = emb_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user_id = str(row["user_id"])
        user_idx = self.user_to_idx[user_id]

        paper_idx = int(row["row_idx"])
        item_vec = self.paper_embeddings[paper_idx].astype(np.float32)

        profile = self.user_profiles.get(user_id, None)
        if profile is None:
            liked_vec = np.zeros(self.emb_dim, dtype=np.float32)
            disliked_vec = np.zeros(self.emb_dim, dtype=np.float32)
        else:
            liked_vec = profile["liked"]
            disliked_vec = profile["disliked"]

        label = np.float32(row["label"])

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "liked_vec": torch.tensor(liked_vec, dtype=torch.float32),
            "disliked_vec": torch.tensor(disliked_vec, dtype=torch.float32),
            "item_vec": torch.tensor(item_vec, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }

# =========================
# Model
# =========================

class TwoTower(nn.Module):
    def __init__(self, num_users, emb_dim, user_id_dim=64, hidden_dim=256, out_dim=128):
        super().__init__()

        self.user_id_emb = nn.Embedding(num_users, user_id_dim)

        self.user_tower = nn.Sequential(
            nn.Linear(user_id_dim + emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

        self.item_tower = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, user_idx, liked_vec, disliked_vec, item_vec):
        uid_vec = self.user_id_emb(user_idx)

        user_input = torch.cat([uid_vec, liked_vec, disliked_vec], dim=1)

        user_z = self.user_tower(user_input)
        item_z = self.item_tower(item_vec)

        user_z = nn.functional.normalize(user_z, dim=1)
        item_z = nn.functional.normalize(item_z, dim=1)

        score = (user_z * item_z).sum(dim=1)
        return score

# =========================
# Train / Eval
# =========================

def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_scores = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            user_idx = batch["user_idx"].to(device)
            liked_vec = batch["liked_vec"].to(device)
            disliked_vec = batch["disliked_vec"].to(device)
            item_vec = batch["item_vec"].to(device)
            label = batch["label"].to(device)

            score = model(user_idx, liked_vec, disliked_vec, item_vec)
            loss = criterion(score, label)

            total_loss += loss.item() * len(label)

            all_labels.extend(label.cpu().numpy().tolist())
            all_scores.extend(score.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)

    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = float("nan")

    return avg_loss, auc

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    paper_embeddings = np.load(EMB_PATH).astype("float32")
    emb_dim = paper_embeddings.shape[1]

    train_df = pd.read_csv(TRAIN_PATH, dtype={"user_id": str}, low_memory=False)
    valid_df = pd.read_csv(VALID_PATH, dtype={"user_id": str}, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, dtype={"user_id": str}, low_memory=False)

    train_df["label"] = train_df["label"].astype(int)
    valid_df["label"] = valid_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_df["row_idx"] = train_df["row_idx"].astype(int)
    valid_df["row_idx"] = valid_df["row_idx"].astype(int)
    test_df["row_idx"] = test_df["row_idx"].astype(int)

    print("paper_embeddings:", paper_embeddings.shape)
    print("train row_idx min/max:", train_df["row_idx"].min(), train_df["row_idx"].max())
    print("valid row_idx min/max:", valid_df["row_idx"].min(), valid_df["row_idx"].max())
    print("test row_idx min/max :", test_df["row_idx"].min(), test_df["row_idx"].max())

    assert train_df["row_idx"].min() >= 0
    assert valid_df["row_idx"].min() >= 0
    assert test_df["row_idx"].min() >= 0
    assert train_df["row_idx"].max() < len(paper_embeddings)
    assert valid_df["row_idx"].max() < len(paper_embeddings)
    assert test_df["row_idx"].max() < len(paper_embeddings)

    # Use only users seen in train
    train_users = sorted(train_df["user_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(train_users)}

    valid_df = valid_df[valid_df["user_id"].isin(user_to_idx)].copy()
    test_df = test_df[test_df["user_id"].isin(user_to_idx)].copy()

    print("train:", len(train_df), "users:", train_df["user_id"].nunique())
    print("valid:", len(valid_df), "users:", valid_df["user_id"].nunique())
    print("test :", len(test_df), "users:", test_df["user_id"].nunique())

    user_profiles = build_user_profiles(train_df, paper_embeddings, emb_dim)

    train_ds = PaperRatingDataset(train_df, paper_embeddings, user_to_idx, user_profiles, emb_dim)
    valid_ds = PaperRatingDataset(valid_df, paper_embeddings, user_to_idx, user_profiles, emb_dim)
    test_ds = PaperRatingDataset(test_df, paper_embeddings, user_to_idx, user_profiles, emb_dim)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TwoTower(
        num_users=len(user_to_idx),
        emb_dim=emb_dim,
        user_id_dim=USER_ID_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_valid_auc = -1.0
    best_path = os.path.join(SAVE_DIR, "best_two_tower_arxividx.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            user_idx = batch["user_idx"].to(device)
            liked_vec = batch["liked_vec"].to(device)
            disliked_vec = batch["disliked_vec"].to(device)
            item_vec = batch["item_vec"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()

            score = model(user_idx, liked_vec, disliked_vec, item_vec)
            loss = criterion(score, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label)

        train_loss = total_loss / len(train_loader.dataset)
        valid_loss, valid_auc = evaluate(model, valid_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | "
            f"valid_auc={valid_auc:.4f}"
        )

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "user_to_idx": user_to_idx,
                    "emb_dim": emb_dim,
                    "config": {
                        "user_id_dim": USER_ID_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "out_dim": OUT_DIM,
                    },
                },
                best_path,
            )
            print("saved best:", best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_auc = evaluate(model, test_loader, device)
    print("=" * 50)
    print(f"Best valid AUC: {best_valid_auc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test AUC : {test_auc:.4f}")

if __name__ == "__main__":
    main()