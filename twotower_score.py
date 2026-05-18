import os
import json
import numpy as np
import pandas as pd
import faiss

from pathlib import Path
import os

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# ============================================================
# Base Path
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# Config & API Key Setup (.env 로드)
# ============================================================
load_dotenv(BASE_DIR / ".env")

gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY가 .env 파일에 설정되어 있지 않습니다."
    )

genai.configure(api_key=gemini_api_key)

# ============================================================
# Data Paths
# ============================================================
PAPER_MAP_PATH = BASE_DIR / "data" / "paper_id_map.csv"
PAPER_EMB_PATH = BASE_DIR / "data" / "paper_embeddings.npy"
FAISS_INDEX_PATH = BASE_DIR / "data" / "papers_faiss.index"

# ============================================================
# Model / Train Paths
# ============================================================
TWO_TOWER_CKPT_PATH = (
    BASE_DIR
    / "two_tower"
    / "checkpoints"
    / "best_two_tower_arxividx.pt"
)

TRAIN_PATH = (
    BASE_DIR
    / "two_tower"
    / "data"
    / "train_arxividx.csv"
)

# ============================================================
# Device
# ============================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
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

assert paper_embeddings.shape[1] == emb_dim, "Embedding dimension mismatch"

two_tower = TwoTower(
    num_users=len(user_to_idx),
    emb_dim=emb_dim,
    user_id_dim=config["user_id_dim"],
    hidden_dim=config["hidden_dim"],
    out_dim=config["out_dim"],
).to(DEVICE)

two_tower.load_state_dict(ckpt["model_state_dict"])
two_tower.eval()

# ============================================================
# Build user profiles from train_arxividx.csv
# ============================================================
train_df = pd.read_csv(TRAIN_PATH, dtype={"user_id": str}, low_memory=False)
train_df["user_id"] = train_df["user_id"].astype(str)

def build_user_profiles(train_df, paper_embeddings, emb_dim):
    profiles = {}
    for user_id, g in train_df.groupby("user_id"):
        liked_idx = g.loc[g["label"] == 1, "row_idx"].astype(int).values
        disliked_idx = g.loc[g["label"] == 0, "row_idx"].astype(int).values

        liked_vec = (
            paper_embeddings[liked_idx].mean(axis=0)
            if len(liked_idx) > 0
            else np.zeros(emb_dim, dtype=np.float32)
        )
        disliked_vec = (
            paper_embeddings[disliked_idx].mean(axis=0)
            if len(disliked_idx) > 0
            else np.zeros(emb_dim, dtype=np.float32)
        )

        profiles[str(user_id)] = {
            "liked": liked_vec.astype(np.float32),
            "disliked": disliked_vec.astype(np.float32),
        }
    return profiles

user_profiles = build_user_profiles(train_df, paper_embeddings, emb_dim)

# ============================================================
# Retrieval functions
# ============================================================
def build_text(title, abstract):
    title = "" if pd.isna(title) else str(title).strip()
    abstract = "" if pd.isna(abstract) else str(abstract).strip()
    return (title + " " + abstract).strip()

def encode_text(text):
    return encoder.encode(
        [text], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

def search_index(query_vec, top_k=100, exclude_paper_id=None):
    scores, indices = index.search(query_vec, top_k + 10)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = paper_map.iloc[idx]
        paper_id = str(row["paper_id"])
        if exclude_paper_id and paper_id == str(exclude_paper_id):
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
    # [FIX] seed 입력 유효성 검사
    if input_type in ("seed", "seed_query"):
        seed_text = build_text(seed_title, seed_abstract)
        if not seed_text:
            raise ValueError("seed 모드에서는 seed_title 또는 seed_abstract가 필요합니다.")

    if input_type == "seed":
        seed_vec = encode_text(build_text(seed_title, seed_abstract))
        return search_index(seed_vec, top_k=top_k, exclude_paper_id=exclude_paper_id)

    elif input_type == "query":
        if not query_text:
            raise ValueError("query 모드에서는 query_text가 필요합니다.")
        return search_index(encode_text(query_text), top_k=top_k)

    elif input_type == "seed_query":
        if not query_text:
            raise ValueError("seed_query 모드에서는 query_text가 필요합니다.")
        seed_vec = encode_text(build_text(seed_title, seed_abstract))
        query_vec = encode_text(query_text)
        final_query_vec = alpha * query_vec + beta * seed_vec
        final_query_vec = final_query_vec / (
            np.linalg.norm(final_query_vec, axis=1, keepdims=True) + 1e-8
        )
        return search_index(
            final_query_vec.astype("float32"),
            top_k=top_k,
            exclude_paper_id=exclude_paper_id,
        )

    else:
        raise ValueError(f"지원하지 않는 input_type: {input_type}")

# ============================================================
# Two-tower scoring & Late Fusion
# ============================================================
def score_with_two_tower(user_id, candidates_df, batch_size=512):
    user_id = str(user_id)

    # [FIX] 알 수 없는 user_id에 대한 예외 처리
    if user_id not in user_to_idx:
        raise ValueError(f"알 수 없는 user_id (user_to_idx에 없음): {user_id}")
    if user_id not in user_profiles:
        raise ValueError(f"알 수 없는 user_id (user_profiles에 없음): {user_id}")

    user_idx = user_to_idx[user_id]
    profile = user_profiles[user_id]
    liked_vec, disliked_vec = profile["liked"], profile["disliked"]

    df = candidates_df.copy()
    item_vecs = paper_embeddings[df["row_idx"].astype(int).values].astype("float32")
    all_scores = []

    with torch.no_grad():
        for start in range(0, len(item_vecs), batch_size):
            end = start + batch_size
            batch_items = item_vecs[start:end]
            bs = len(batch_items)

            user_idx_tensor = torch.full(
                (bs,), fill_value=user_idx, dtype=torch.long, device=DEVICE
            )
            liked_tensor = torch.tensor(
                np.repeat(liked_vec[None, :], bs, axis=0), dtype=torch.float32, device=DEVICE
            )
            disliked_tensor = torch.tensor(
                np.repeat(disliked_vec[None, :], bs, axis=0), dtype=torch.float32, device=DEVICE
            )
            item_tensor = torch.tensor(batch_items, dtype=torch.float32, device=DEVICE)

            scores = two_tower(user_idx_tensor, liked_tensor, disliked_tensor, item_tensor)
            all_scores.extend(scores.cpu().numpy().tolist())

    df["two_tower_score"] = all_scores
    return df

def minmax_norm(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    x_min, x_max = x.min(), x.max()
    return (
        np.ones_like(x) * 0.5
        if abs(x_max - x_min) < 1e-8
        else (x - x_min) / (x_max - x_min)
    )

def rerank(scored_df, semantic_weight=0.5, two_tower_weight=0.5):
    df = scored_df.copy()
    df["semantic_score_norm"] = minmax_norm(df["semantic_score"].values)
    df["two_tower_score_norm"] = minmax_norm(df["two_tower_score"].values)

    total = semantic_weight + two_tower_weight
    df["final_score"] = (
        (semantic_weight / total) * df["semantic_score_norm"]
        + (two_tower_weight / total) * df["two_tower_score_norm"]
    )
    return df.sort_values("final_score", ascending=False).reset_index(drop=True)

# ============================================================
# Recency-aware Adjustment
# ============================================================
def apply_time_decay(df, lambda_val=0.05):
    df = df.copy()
    df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")

    # 현재 날짜로 동적 처리
    current_date = pd.Timestamp.now()

    df["months_passed"] = (
        (current_date.year - df["update_date"].dt.year) * 12
        + (current_date.month - df["update_date"].dt.month)
    )
    df["months_passed"] = df["months_passed"].clip(lower=0)

    df["time_penalty"] = np.exp(-lambda_val * df["months_passed"])
    df["time_adjusted_score"] = df["final_score"] * df["time_penalty"]

    df = df.sort_values("time_adjusted_score", ascending=False).reset_index(drop=True)
    return df

# ============================================================
# LLM User Agent Reranking (Gemini)
# ============================================================
def llm_user_agent_rerank(user_query, top_candidates_df, final_k=10):
    model = genai.GenerativeModel("gemini-2.0-flash")

    papers_info = ""
    for _, row in top_candidates_df.head(20).iterrows():
        papers_info += (
            f"[Paper ID: {row['paper_id']}]\n"
            f"Title: {row['title']}\n"
            f"Update Date: {row['update_date']}\n---\n"
        )

    prompt = f"""
You are an AI Research Assistant. The user wants papers about: "{user_query}"

Here is a list of top candidate papers ranked by our internal system:
{papers_info}

Task:
1. Select the Top {final_k} papers from the list that best match the user's intent.
2. Provide a brief 1-sentence reason WHY each paper is recommended for this user.

You MUST output the result strictly as a valid JSON array like this:
[
  {{"paper_id": "1234.5678", "reason": "This paper is highly relevant because..."}}
]
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )

        reranked_list = json.loads(response.text)
        final_results = []
        for rank, item in enumerate(reranked_list):
            paper_id = item.get("paper_id")
            reason = item.get("reason")
            matched_rows = top_candidates_df[top_candidates_df["paper_id"] == paper_id]
            if not matched_rows.empty:
                paper_row = matched_rows.iloc[0]
                final_results.append({
                    "final_rank": rank + 1,
                    "paper_id": paper_id,
                    "title": paper_row["title"],
                    "reason": reason,
                    "time_adjusted_score": paper_row["time_adjusted_score"],
                })

        # 매칭 결과가 없을 경우 기존 랭킹으로 fallback
        if not final_results:
            print("경고: LLM 재랭킹 결과가 비어 있어 기존 랭킹으로 fallback합니다.")
            fallback_df = top_candidates_df.head(final_k).copy()
            fallback_df["final_rank"] = range(1, len(fallback_df) + 1)
            fallback_df["reason"] = "LLM 응답을 파싱하지 못해 기존 추천 순위를 반환합니다."
            return fallback_df

        return pd.DataFrame(final_results).head(final_k)

    except Exception as e:
        print(f"Gemini API 오류: {e}. 기존 랭킹으로 fallback합니다.")
        fallback_df = top_candidates_df.head(final_k).copy()
        fallback_df["final_rank"] = range(1, len(fallback_df) + 1)
        fallback_df["reason"] = "API 호출 오류로 인해 기존 추천 순위를 임시 반환합니다."
        return fallback_df

# ============================================================
# Final Pipeline (Recommend)
# ============================================================
def recommend(
    user_id,
    input_type,
    user_query="",
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
    lambda_val=0.05,
):
    # 1. Retrieval
    candidates = retrieve_candidates(
        input_type, candidate_k, seed_title, seed_abstract,
        query_text, exclude_paper_id, alpha, beta,
    )
    # 2. Two-Tower Scoring
    scored = score_with_two_tower(user_id, candidates)
    # 3. Late Fusion Rerank
    ranked = rerank(scored, semantic_weight, two_tower_weight)
    # 4. Time Decay
    time_adjusted = apply_time_decay(ranked, lambda_val)
    # 5. LLM User Agent Rerank
    actual_query = user_query or query_text or seed_title or ""
    final_df = llm_user_agent_rerank(actual_query, time_adjusted, final_k)

    return final_df

# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    user_id = str(train_df["user_id"].value_counts().index[0])

    print("\n=== Running Pipeline (Seed + Query) ===")

    seed_title = "RETHINKING ATTENTION WITH PERFORMERS"
    seed_abstract = (
        "We introduce Performers, Transformer architectures which can estimate regular "
        "softmax full-rank-attention Transformers with provable accuracy, but using only "
        "linear space and time complexity."
    )
    query_text = "long sequence attention mechanism"

    result = recommend(
        user_id=user_id,
        input_type="seed_query",
        user_query="I want an efficient attention mechanism for long sequences, similar to Performers.",
        seed_title=seed_title,
        seed_abstract=seed_abstract,
        query_text=query_text,
        lambda_val=0.1,
        candidate_k=100,
        final_k=5,
    )

    print("\n[최종 추천 결과]")
    pd.set_option("display.max_colwidth", None)
    print(result[["final_rank", "paper_id", "title", "time_adjusted_score", "reason"]])
