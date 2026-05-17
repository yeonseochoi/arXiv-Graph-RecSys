import os
import numpy as np
import pandas as pd
import faiss

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


# ============================================================
# Config
# ============================================================

PAPER_MAP_PATH = "paper_id_map.csv"
PAPER_EMB_PATH = "paper_embeddings.npy"
FAISS_INDEX_PATH = "papers_faiss.index"

TWO_TOWER_CKPT_PATH = "two_tower/checkpoints/best_two_tower_arxividx.pt"
TRAIN_PATH = "two_tower/data/train_arxividx.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Load retrieval artifacts
# ============================================================

paper_map = pd.read_csv(PAPER_MAP_PATH)
paper_map["paper_id"] = paper_map["paper_id"].astype(str)

paper_embeddings = np.load(PAPER_EMB_PATH).astype("float32")
index = faiss.read_index(FAISS_INDEX_PATH)

encoder = SentenceTransformer("sentence-transformers/allenai-specter")

print("paper_map:", paper_map.shape)
print("paper_embeddings:", paper_embeddings.shape)
print("faiss ntotal:", index.ntotal)
print("device:", DEVICE)


# ============================================================
# TwoTower model
# ============================================================

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


# ============================================================
# Load checkpoint
# ============================================================

ckpt = torch.load(TWO_TOWER_CKPT_PATH, map_location=DEVICE)

user_to_idx = ckpt["user_to_idx"]
config = ckpt["config"]
emb_dim = ckpt["emb_dim"]

assert paper_embeddings.shape[1] == emb_dim, (
    f"paper embedding dim {paper_embeddings.shape[1]} != checkpoint emb_dim {emb_dim}"
)

two_tower = TwoTower(
    num_users=len(user_to_idx),
    emb_dim=emb_dim,
    user_id_dim=config["user_id_dim"],
    hidden_dim=config["hidden_dim"],
    out_dim=config["out_dim"],
).to(DEVICE)

two_tower.load_state_dict(ckpt["model_state_dict"])
two_tower.eval()

print("two-tower loaded.")
print("num users:", len(user_to_idx))


# ============================================================
# Build user profiles from train_arxividx.csv
# ============================================================

train_df = pd.read_csv(TRAIN_PATH, dtype={"user_id": str}, low_memory=False)
train_df["user_id"] = train_df["user_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)
train_df["row_idx"] = train_df["row_idx"].astype(int)

print("train row_idx min/max:", train_df["row_idx"].min(), train_df["row_idx"].max())
print("embedding max idx:", len(paper_embeddings) - 1)

assert train_df["row_idx"].min() >= 0
assert train_df["row_idx"].max() < len(paper_embeddings)


def build_user_profiles(train_df, paper_embeddings, emb_dim):
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

        profiles[str(user_id)] = {
            "liked": liked_vec.astype(np.float32),
            "disliked": disliked_vec.astype(np.float32),
        }

    return profiles


user_profiles = build_user_profiles(train_df, paper_embeddings, emb_dim)
print("user profiles:", len(user_profiles))


# ============================================================
# Retrieval functions: 3 input cases
# ============================================================

def build_text(title, abstract):
    title = "" if pd.isna(title) else str(title).strip()
    abstract = "" if pd.isna(abstract) else str(abstract).strip()
    return (title + " " + abstract).strip()


def encode_text(text):
    vec = encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return vec


def search_index(query_vec, top_k=100, exclude_paper_id=None):
    scores, indices = index.search(query_vec, top_k + 10)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = paper_map.iloc[idx]
        paper_id = str(row["paper_id"])

        if exclude_paper_id is not None and paper_id == str(exclude_paper_id):
            continue

        results.append({
            "row_idx": int(row["row_idx"]),
            "paper_id": paper_id,
            "title": row["title"],
            "update_date": row["update_date"],
            "categories": row["categories"],
            "semantic_score": float(score),
        })

        if len(results) == top_k:
            break

    return pd.DataFrame(results)


def retrieve_by_seed_paper(seed_title, seed_abstract, top_k=100, exclude_paper_id=None):
    seed_text = build_text(seed_title, seed_abstract)
    seed_vec = encode_text(seed_text)

    return search_index(
        seed_vec,
        top_k=top_k,
        exclude_paper_id=exclude_paper_id,
    )


def retrieve_by_query(query_text, top_k=100):
    query_vec = encode_text(query_text)

    return search_index(
        query_vec,
        top_k=top_k,
    )


def retrieve_by_seed_and_query(
    seed_title,
    seed_abstract,
    query_text,
    alpha=0.7,
    beta=0.3,
    top_k=100,
    exclude_paper_id=None,
):
    seed_text = build_text(seed_title, seed_abstract)

    seed_vec = encode_text(seed_text)
    query_vec = encode_text(query_text)

    final_query_vec = alpha * query_vec + beta * seed_vec
    final_query_vec = final_query_vec / (
        np.linalg.norm(final_query_vec, axis=1, keepdims=True) + 1e-8
    )
    final_query_vec = final_query_vec.astype("float32")

    return search_index(
        final_query_vec,
        top_k=top_k,
        exclude_paper_id=exclude_paper_id,
    )


def retrieve_candidates(
    input_type,
    top_k=100,
    seed_title=None,
    seed_abstract=None,
    query_text=None,
    exclude_paper_id=None,
    alpha=0.7,
    beta=0.3,
):
    """
    input_type:
        - "seed"
        - "query"
        - "seed_query"
    """

    if input_type == "seed":
        if seed_title is None or seed_abstract is None:
            raise ValueError("input_type='seed' requires seed_title and seed_abstract.")

        return retrieve_by_seed_paper(
            seed_title=seed_title,
            seed_abstract=seed_abstract,
            top_k=top_k,
            exclude_paper_id=exclude_paper_id,
        )

    if input_type == "query":
        if query_text is None:
            raise ValueError("input_type='query' requires query_text.")

        return retrieve_by_query(
            query_text=query_text,
            top_k=top_k,
        )

    if input_type == "seed_query":
        if seed_title is None or seed_abstract is None or query_text is None:
            raise ValueError(
                "input_type='seed_query' requires seed_title, seed_abstract, and query_text."
            )

        return retrieve_by_seed_and_query(
            seed_title=seed_title,
            seed_abstract=seed_abstract,
            query_text=query_text,
            alpha=alpha,
            beta=beta,
            top_k=top_k,
            exclude_paper_id=exclude_paper_id,
        )

    raise ValueError("input_type must be one of: 'seed', 'query', 'seed_query'.")


# ============================================================
# Two-tower scoring
# ============================================================

def score_with_two_tower(user_id, candidates_df, batch_size=512):
    user_id = str(user_id)

    if user_id not in user_to_idx:
        raise ValueError(
            f"user_id={user_id} not found in trained users. "
            "Use an existing user_id from train_arxividx.csv."
        )

    if user_id not in user_profiles:
        raise ValueError(f"user_id={user_id} has no user profile.")

    df = candidates_df.copy()

    user_idx = user_to_idx[user_id]
    profile = user_profiles[user_id]

    liked_vec = profile["liked"]
    disliked_vec = profile["disliked"]

    row_indices = df["row_idx"].astype(int).values

    assert row_indices.min() >= 0
    assert row_indices.max() < len(paper_embeddings)

    item_vecs = paper_embeddings[row_indices].astype("float32")

    all_scores = []

    two_tower.eval()

    with torch.no_grad():
        for start in range(0, len(item_vecs), batch_size):
            end = start + batch_size
            batch_items = item_vecs[start:end]
            bs = len(batch_items)

            user_idx_tensor = torch.full(
                (bs,),
                fill_value=user_idx,
                dtype=torch.long,
                device=DEVICE,
            )

            liked_tensor = torch.tensor(
                np.repeat(liked_vec[None, :], bs, axis=0),
                dtype=torch.float32,
                device=DEVICE,
            )

            disliked_tensor = torch.tensor(
                np.repeat(disliked_vec[None, :], bs, axis=0),
                dtype=torch.float32,
                device=DEVICE,
            )

            item_tensor = torch.tensor(
                batch_items,
                dtype=torch.float32,
                device=DEVICE,
            )

            scores = two_tower(
                user_idx_tensor,
                liked_tensor,
                disliked_tensor,
                item_tensor,
            )

            all_scores.extend(scores.cpu().numpy().tolist())

    df["two_tower_score"] = all_scores
    return df


# ============================================================
# Final ranking: semantic + two-tower only
# ============================================================

def minmax_norm(x):
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return x

    x_min = x.min()
    x_max = x.max()

    if abs(x_max - x_min) < 1e-8:
        return np.ones_like(x) * 0.5

    return (x - x_min) / (x_max - x_min)


def rerank(scored_df, semantic_weight=0.5, two_tower_weight=0.5):
    df = scored_df.copy()

    df["semantic_score_norm"] = minmax_norm(df["semantic_score"].values)
    df["two_tower_score_norm"] = minmax_norm(df["two_tower_score"].values)

    total = semantic_weight + two_tower_weight
    if total <= 0:
        raise ValueError("semantic_weight + two_tower_weight must be positive.")

    semantic_weight = semantic_weight / total
    two_tower_weight = two_tower_weight / total

    df["final_score"] = (
        semantic_weight * df["semantic_score_norm"]
        + two_tower_weight * df["two_tower_score_norm"]
    )

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    return df


# ============================================================
# One-shot recommendation
# ============================================================

def recommend(
    user_id,
    input_type,
    seed_title=None,
    seed_abstract=None,
    query_text=None,
    exclude_paper_id=None,
    candidate_k=100,
    final_k=10,
    alpha=0.7,
    beta=0.3,
    semantic_weight=0.5,
    two_tower_weight=0.5,
):
    candidates = retrieve_candidates(
        input_type=input_type,
        top_k=candidate_k,
        seed_title=seed_title,
        seed_abstract=seed_abstract,
        query_text=query_text,
        exclude_paper_id=exclude_paper_id,
        alpha=alpha,
        beta=beta,
    )

    scored = score_with_two_tower(
        user_id=user_id,
        candidates_df=candidates,
    )

    ranked = rerank(
        scored,
        semantic_weight=semantic_weight,
        two_tower_weight=two_tower_weight,
    )

    return ranked.head(final_k)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    print("\nAvailable user examples:")
    print(train_df["user_id"].value_counts().head())

    user_id = str(train_df["user_id"].value_counts().index[0])

    # Case 1. query only
    result = recommend(
        user_id=user_id,
        input_type="query",
        query_text="long sequence attention mechanism",
        candidate_k=100,
        final_k=10,
        semantic_weight=0.5,
        two_tower_weight=0.5,
    )

    print("\n=== Query Only Recommendation ===")
    print(result[
        [
            "rank",
            "paper_id",
            "title",
            "semantic_score",
            "two_tower_score",
            "final_score",
        ]
    ])

    # Case 2. seed only
    seed_title = "RETHINKING ATTENTION WITH PERFORMERS"
    seed_abstract = (
        "We introduce Performers, Transformer architectures which can estimate regular "
        "softmax full-rank-attention Transformers with provable accuracy, but using only "
        "linear space and time complexity."
    )

    result = recommend(
        user_id=user_id,
        input_type="seed",
        seed_title=seed_title,
        seed_abstract=seed_abstract,
        candidate_k=100,
        final_k=10,
    )

    print("\n=== Seed Paper Recommendation ===")
    print(result[
        [
            "rank",
            "paper_id",
            "title",
            "semantic_score",
            "two_tower_score",
            "final_score",
        ]
    ])

    # Case 3. seed + query
    result = recommend(
        user_id=user_id,
        input_type="seed_query",
        seed_title=seed_title,
        seed_abstract=seed_abstract,
        query_text="long sequence attention mechanism",
        alpha=0.7,
        beta=0.3,
        candidate_k=100,
        final_k=10,
    )

    print("\n=== Seed + Query Recommendation ===")
    print(result[
        [
            "rank",
            "paper_id",
            "title",
            "semantic_score",
            "two_tower_score",
            "final_score",
        ]
    ])