import argparse
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Seed
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Two-Tower Model
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

        return (user_z * item_z).sum(dim=1)


# ============================================================
# Utils
# ============================================================

def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype("float32", copy=False)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return x

    mn = float(np.min(x))
    mx = float(np.max(x))

    if abs(mx - mn) < 1e-8:
        return np.ones_like(x, dtype=np.float32) * 0.5

    return (x - mn) / (mx - mn)


def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"user_id": str}, low_memory=False)

    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)

    if "paper_id" in df.columns:
        df["paper_id"] = df["paper_id"].astype(str)

    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    if "row_idx" in df.columns:
        df["row_idx"] = df["row_idx"].astype(int)

    return df


def load_faiss_index(index_path: Optional[str]):
    if not index_path:
        return None

    if not os.path.exists(index_path):
        print(f"[WARN] FAISS index not found: {index_path}")
        print("[WARN] brute-force search를 사용합니다.")
        return None

    try:
        import faiss
        return faiss.read_index(index_path)
    except Exception as e:
        print(f"[WARN] FAISS load failed: {e}")
        print("[WARN] brute-force search를 사용합니다.")
        return None


# ============================================================
# User profile
# ============================================================

def build_user_profiles(
    train_df: pd.DataFrame,
    paper_embeddings: np.ndarray,
    emb_dim: int,
) -> Dict[str, Dict[str, np.ndarray]]:

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


def build_train_interacted_map(train_df: pd.DataFrame) -> Dict[str, set]:
    interacted = {}

    for uid, g in train_df.groupby("user_id"):
        interacted[str(uid)] = set(g["row_idx"].astype(int).tolist())

    return interacted


def get_eligible_users(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    min_train_liked: int = 2,
    min_test_liked: int = 1,
    max_users: Optional[int] = None,
    seed: int = 42,
) -> List[str]:

    train_liked = (
        train_df[train_df["label"] == 1]
        .groupby("user_id")["row_idx"]
        .count()
    )

    test_liked = (
        test_df[test_df["label"] == 1]
        .groupby("user_id")["row_idx"]
        .count()
    )

    users = sorted(
        set(train_liked[train_liked >= min_train_liked].index.astype(str))
        & set(test_liked[test_liked >= min_test_liked].index.astype(str))
        & set(user_to_idx.keys())
    )

    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = sorted(rng.choice(users, size=max_users, replace=False).tolist())

    return users


# ============================================================
# Candidate retrieval
# ============================================================

def retrieve_by_profile_vector(
    user_id: str,
    user_profiles: Dict[str, Dict[str, np.ndarray]],
    paper_embeddings_norm: np.ndarray,
    paper_map: pd.DataFrame,
    faiss_index,
    top_k: int,
    exclude_row_idxs: Optional[set] = None,
) -> pd.DataFrame:

    exclude_row_idxs = exclude_row_idxs or set()

    query_vec = user_profiles[user_id]["liked"].astype("float32")
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    query_vec_2d = query_vec.reshape(1, -1).astype("float32")

    search_k = min(len(paper_map), top_k + len(exclude_row_idxs) + 200)

    if faiss_index is not None:
        scores, indices = faiss_index.search(query_vec_2d, search_k)

        rows = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            row = paper_map.iloc[int(idx)].copy()
            row_idx = int(row["row_idx"])

            if row_idx in exclude_row_idxs:
                continue

            row["semantic_score"] = float(score)
            rows.append(row)

            if len(rows) >= top_k:
                break

        return pd.DataFrame(rows).reset_index(drop=True)

    scores = paper_embeddings_norm @ query_vec
    order = np.argsort(-scores)

    rows = []

    for idx in order:
        row = paper_map.iloc[int(idx)].copy()
        row_idx = int(row["row_idx"])

        if row_idx in exclude_row_idxs:
            continue

        row["semantic_score"] = float(scores[int(idx)])
        rows.append(row)

        if len(rows) >= top_k:
            break

    return pd.DataFrame(rows).reset_index(drop=True)


def retrieve_by_query_text(
    query_text: str,
    encoder,
    paper_embeddings_norm: np.ndarray,
    paper_map: pd.DataFrame,
    faiss_index,
    top_k: int,
) -> pd.DataFrame:

    query_vec = encoder.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    search_k = min(len(paper_map), top_k)

    if faiss_index is not None:
        scores, indices = faiss_index.search(query_vec, search_k)

        rows = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            row = paper_map.iloc[int(idx)].copy()
            row["semantic_score"] = float(score)
            rows.append(row)

        return pd.DataFrame(rows).reset_index(drop=True)

    q = query_vec[0]
    scores = paper_embeddings_norm @ q
    order = np.argsort(-scores)[:search_k]

    rows = []

    for idx in order:
        row = paper_map.iloc[int(idx)].copy()
        row["semantic_score"] = float(scores[int(idx)])
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================
# Scoring / Ranking
# ============================================================

def score_two_tower(
    model: TwoTower,
    user_id: str,
    candidates_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    user_profiles: Dict[str, Dict[str, np.ndarray]],
    paper_embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> pd.DataFrame:

    df = candidates_df.copy()

    uid_idx = user_to_idx[str(user_id)]
    profile = user_profiles[str(user_id)]

    liked_vec = profile["liked"]
    disliked_vec = profile["disliked"]

    item_vecs = paper_embeddings[df["row_idx"].astype(int).values].astype("float32")

    all_scores = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(item_vecs), batch_size):
            end = start + batch_size
            batch_items = item_vecs[start:end]
            bs = len(batch_items)

            user_idx_t = torch.full(
                (bs,),
                fill_value=uid_idx,
                dtype=torch.long,
                device=device,
            )

            liked_t = torch.tensor(
                np.repeat(liked_vec[None, :], bs, axis=0),
                dtype=torch.float32,
                device=device,
            )

            disliked_t = torch.tensor(
                np.repeat(disliked_vec[None, :], bs, axis=0),
                dtype=torch.float32,
                device=device,
            )

            item_t = torch.tensor(
                batch_items,
                dtype=torch.float32,
                device=device,
            )

            score = model(user_idx_t, liked_t, disliked_t, item_t)
            all_scores.extend(score.cpu().numpy().tolist())

    df["two_tower_score"] = all_scores

    return df


def apply_late_fusion(
    scored_df: pd.DataFrame,
    semantic_weight: float,
    two_tower_weight: float,
) -> pd.DataFrame:

    df = scored_df.copy()

    if "two_tower_score" not in df.columns:
        df["two_tower_score"] = 0.0

    df["semantic_score_norm"] = minmax_norm(df["semantic_score"].values)
    df["two_tower_score_norm"] = minmax_norm(df["two_tower_score"].values)

    total = semantic_weight + two_tower_weight

    if total <= 0:
        raise ValueError("semantic_weight + two_tower_weight must be > 0")

    df["final_score"] = (
        (semantic_weight / total) * df["semantic_score_norm"]
        + (two_tower_weight / total) * df["two_tower_score_norm"]
    )

    return df.sort_values("final_score", ascending=False).reset_index(drop=True)


def apply_recency(
    ranked_df: pd.DataFrame,
    lambda_val: float,
    current_date: str,
) -> pd.DataFrame:

    df = ranked_df.copy()

    if lambda_val <= 0:
        df["time_adjusted_score"] = df["final_score"]
        return df.sort_values("time_adjusted_score", ascending=False).reset_index(drop=True)

    df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")
    cur = pd.Timestamp(current_date)

    months_passed = (
        (cur.year - df["update_date"].dt.year) * 12
        + (cur.month - df["update_date"].dt.month)
    )

    df["months_passed"] = months_passed.clip(lower=0).fillna(9999)
    df["time_penalty"] = np.exp(-lambda_val * df["months_passed"])
    df["time_adjusted_score"] = df["final_score"] * df["time_penalty"]

    return df.sort_values("time_adjusted_score", ascending=False).reset_index(drop=True)


def recommend_for_user(
    user_id: str,
    candidate_cache: Dict[str, pd.DataFrame],
    model: TwoTower,
    user_to_idx: Dict[str, int],
    user_profiles: Dict[str, Dict[str, np.ndarray]],
    paper_embeddings: np.ndarray,
    device: torch.device,
    semantic_weight: float,
    two_tower_weight: float,
    lambda_val: float,
    current_date: str,
    final_k: int = 20,
    batch_size: int = 512,
) -> pd.DataFrame:

    candidates = candidate_cache[user_id]

    scored = score_two_tower(
        model=model,
        user_id=user_id,
        candidates_df=candidates,
        user_to_idx=user_to_idx,
        user_profiles=user_profiles,
        paper_embeddings=paper_embeddings,
        device=device,
        batch_size=batch_size,
    )

    ranked = apply_late_fusion(
        scored_df=scored,
        semantic_weight=semantic_weight,
        two_tower_weight=two_tower_weight,
    )

    ranked = apply_recency(
        ranked_df=ranked,
        lambda_val=lambda_val,
        current_date=current_date,
    )

    return ranked.head(final_k).copy()


# ============================================================
# LLM Reranking
# ============================================================


def build_user_interest_text(
    user_id: str,
    train_df: pd.DataFrame,
    max_titles: int = 5,
) -> str:
    """
    LLM reranking용 user history 요약.
    train interaction 중 liked paper title 일부를 prompt에 넣는다.
    """

    user_id = str(user_id)

    liked = train_df[
        (train_df["user_id"].astype(str) == user_id)
        & (train_df["label"] == 1)
    ].copy()

    if liked.empty:
        return "No explicit liked-paper history is available."

    if "time" in liked.columns:
        liked["time"] = pd.to_datetime(liked["time"], errors="coerce")
        liked = liked.sort_values("time", ascending=False)

    if "title" not in liked.columns:
        return "No liked-paper titles are available."

    titles = (
        liked["title"]
        .dropna()
        .astype(str)
        .head(max_titles)
        .tolist()
    )

    if not titles:
        return "No liked-paper titles are available."

    text = "The user previously liked these papers:\n"

    for i, title in enumerate(titles, 1):
        text += f"{i}. {title}\n"

    return text


def llm_rerank_for_eval(
    user_query: str,
    ranked_df: pd.DataFrame,
    final_k: int = 10,
    model_name: str = "gemini-2.5-flash",
    top_candidates: int = 50,
) -> pd.DataFrame:
    """
    app.py와 동일한 Gemini reranking prompt를 evaluation용으로 사용.
    """

    try:
        import google.generativeai as genai
        from dotenv import load_dotenv

        load_dotenv()
        gemini_key = os.environ.get("GEMINI_API_KEY")

        if not gemini_key:
            print("[WARN] GEMINI_API_KEY가 없어 LLM reranking 없이 시스템 점수 기반 결과를 사용합니다.")
            fallback_df = ranked_df.head(final_k).copy()
            fallback_df["final_rank"] = range(1, len(fallback_df) + 1)
            fallback_df["reason"] = "통합 시스템 점수를 기반으로 추천합니다."
            fallback_df["llm_status"] = "fallback_no_api_key"
            return fallback_df

        genai.configure(api_key=gemini_key)
        model_llm = genai.GenerativeModel("gemini-2.5-flash")

        cands = ranked_df.copy()

        papers_info = ""
        for rank, (_, row) in enumerate(cands.head(top_candidates).iterrows(), start=1):
            papers_info += (
                "[Internal Rank: {}]\n"
                "Paper ID: {}\n"
                "Title: {}\n"
                "Categories: {}\n"
                "Update Date: {}\n"
                "Recency-adjusted Score: {:.6f}\n"
                "---\n"
            ).format(
                rank,
                row.get("paper_id", ""),
                row.get("title", ""),
                row.get("categories", ""),
                row.get("update_date", ""),
                float(row.get("time_adjusted_score", row.get("final_score", 0.0))),
            )

        prompt = f"""
You are an AI Research Assistant. The user wants papers about:
"{user_query}"

Here is a list of candidate papers already ranked by our internal system:
{papers_info}

Task:
1. Select exactly {final_k} unique papers from the candidate list.
2. Treat the existing internal rank and recency-adjusted score as strong prior signals.
3. Change the order significantly only when a paper is clearly more relevant to the user's intent.
4. Provide a polite Korean 1-sentence recommendation reason for each selected paper.
5. Never invent a paper ID and never return the same paper ID twice.

Output strictly as a valid JSON array:
[
{{"paper_id": "1234.5678", "reason": "추천 이유"}}
]
"""

        try:
            resp = model_llm.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )

            reranked = json.loads(resp.text)
            if not isinstance(reranked, list):
                raise ValueError("LLM response must be a JSON array.")

            results = []
            selected_ids = set()

            for item in reranked:
                if not isinstance(item, dict):
                    continue

                pid = str(item.get("paper_id", "")).strip()
                reason = str(item.get("reason", "")).strip()

                if not pid or pid in selected_ids:
                    continue

                matched = cands[cands["paper_id"].astype(str) == pid]
                if matched.empty:
                    continue

                r = matched.iloc[0]
                results.append({
                    "final_rank": len(results) + 1,
                    "paper_id": pid,
                    "title": r["title"],
                    "reason": reason or "통합 시스템 점수를 기반으로 추천합니다.",
                    "update_date": r["update_date"],
                    "categories": r.get("categories", ""),
                    "row_idx": int(r["row_idx"]),
                    "semantic_score": r.get("semantic_score", np.nan),
                    "two_tower_score": r.get("two_tower_score", np.nan),
                    "final_score": r.get("final_score", np.nan),
                    "time_adjusted_score": r.get("time_adjusted_score", np.nan),
                    "llm_status": "success",
                })
                selected_ids.add(pid)

                if len(results) == final_k:
                    break

            # LLM 결과가 부족하면 기존 ranking 순서대로 보충
            for _, r in cands.iterrows():
                pid = str(r["paper_id"])
                if pid in selected_ids:
                    continue

                results.append({
                    "final_rank": len(results) + 1,
                    "paper_id": pid,
                    "title": r["title"],
                    "reason": "통합 시스템 점수를 기반으로 추천합니다.",
                    "update_date": r["update_date"],
                    "categories": r.get("categories", ""),
                    "row_idx": int(r["row_idx"]),
                    "semantic_score": r.get("semantic_score", np.nan),
                    "two_tower_score": r.get("two_tower_score", np.nan),
                    "final_score": r.get("final_score", np.nan),
                    "time_adjusted_score": r.get("time_adjusted_score", np.nan),
                    "llm_status": "success_filled_with_internal_rank",
                })
                selected_ids.add(pid)

                if len(results) == final_k:
                    break

            if not results:
                raise ValueError("No valid LLM-selected papers matched candidate list.")

            return pd.DataFrame(results).head(final_k)

        except Exception as e:
            print(f"[WARN] AI 재정렬 프로세스 일시 지연: {e}")
            fallback_df = cands.head(final_k).copy()
            fallback_df["final_rank"] = range(1, len(fallback_df) + 1)
            fallback_df["reason"] = "네트워크 지연으로 인해 시스템 알고리즘 추천 스코어 기반으로 제공합니다."
            fallback_df["llm_status"] = "fallback_generation_error"
            return fallback_df

    except Exception as e:
        print(f"[WARN] Gemini 설정 또는 호출 준비 중 오류 발생: {e}")
        fallback_df = ranked_df.head(final_k).copy()
        fallback_df["final_rank"] = range(1, len(fallback_df) + 1)
        fallback_df["reason"] = "네트워크 지연으로 인해 시스템 알고리즘 추천 스코어 기반으로 제공합니다."
        fallback_df["llm_status"] = "fallback_setup_error"
        return fallback_df


def run_full_system_with_llm(
    users: List[str],
    candidate_cache: Dict[str, pd.DataFrame],
    model: TwoTower,
    user_to_idx: Dict[str, int],
    user_profiles: Dict[str, Dict[str, np.ndarray]],
    paper_embeddings: np.ndarray,
    device: torch.device,
    gt_by_user: Dict[str, set],
    ks: List[int],
    current_date: str,
    final_k: int,
    batch_size: int,
    llm_model: str,
    llm_top_candidates: int,
    train_df: pd.DataFrame,
    user_query: str = "",
    query_text: str = "",
    seed_title: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:

    recs_by_user = {}

    base_query = (
        user_query
        or query_text
        or seed_title
        or "papers similar to the user's historically liked AI research papers"
    )

    llm_status_counts = {}

    for i, uid in enumerate(users, 1):
        try:
            before_llm = recommend_for_user(
                user_id=uid,
                candidate_cache=candidate_cache,
                model=model,
                user_to_idx=user_to_idx,
                user_profiles=user_profiles,
                paper_embeddings=paper_embeddings,
                device=device,
                semantic_weight=0.5,
                two_tower_weight=0.5,
                lambda_val=0.05,
                current_date=current_date,
                final_k=max(final_k, llm_top_candidates),
                batch_size=batch_size,
            )

            user_history_text = build_user_interest_text(
                user_id=uid,
                train_df=train_df,
                max_titles=5,
            )

            actual_query = f"""{base_query}

{user_history_text}

Please rerank the candidate papers by considering both:
1. the user's current query
2. the user's historically liked papers
"""

            after_llm = llm_rerank_for_eval(
                user_query=actual_query,
                ranked_df=before_llm,
                final_k=final_k,
                model_name=llm_model,
                top_candidates=llm_top_candidates,
            )

            status = (
                after_llm["llm_status"].iloc[0]
                if "llm_status" in after_llm.columns and len(after_llm) > 0
                else "unknown"
            )
            llm_status_counts[status] = llm_status_counts.get(status, 0) + 1

            recs_by_user[uid] = after_llm

        except Exception as e:
            print(f"[WARN] Full system skip user={uid}: {e}")
            llm_status_counts["full_system_exception"] = (
                llm_status_counts.get("full_system_exception", 0) + 1
            )

        if i % 10 == 0:
            print(f"[Full system with LLM] {i}/{len(users)} users")
            print(f"[LLM status counts] {llm_status_counts}")

    print(f"[LLM status counts] {llm_status_counts}")

    summary_df, user_df = evaluate_predictions(
        recs_by_user=recs_by_user,
        gt_by_user=gt_by_user,
        ks=ks,
    )

    summary_df.insert(0, "method", "Full system")
    user_df.insert(0, "method", "Full system")

    for status, count in llm_status_counts.items():
        summary_df[f"llm_{status}"] = count

    return summary_df, user_df, recs_by_user


# ============================================================
# Metrics
# ============================================================

def recall_at_k(pred: List[int], gt: set, k: int) -> float:
    if len(gt) == 0:
        return np.nan

    return len(set(pred[:k]) & gt) / len(gt)


def hitrate_at_k(pred: List[int], gt: set, k: int) -> float:
    if len(gt) == 0:
        return np.nan

    return float(len(set(pred[:k]) & gt) > 0)


def ndcg_at_k(pred: List[int], gt: set, k: int) -> float:
    if len(gt) == 0:
        return np.nan

    dcg = 0.0

    for i, item in enumerate(pred[:k]):
        if item in gt:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return np.nan

    return dcg / idcg


def evaluate_predictions(
    recs_by_user: Dict[str, pd.DataFrame],
    gt_by_user: Dict[str, set],
    ks: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    user_rows = []

    for uid, rec_df in recs_by_user.items():
        gt = gt_by_user.get(uid, set())
        pred = rec_df["row_idx"].astype(int).tolist()

        row = {
            "user_id": uid,
            "num_test_liked": len(gt),
        }

        for k in ks:
            row[f"Recall@{k}"] = recall_at_k(pred, gt, k)
            row[f"NDCG@{k}"] = ndcg_at_k(pred, gt, k)
            row[f"HitRate@{k}"] = hitrate_at_k(pred, gt, k)

        user_rows.append(row)

    user_df = pd.DataFrame(user_rows)

    avg = {}

    for k in ks:
        avg[f"Recall@{k}"] = user_df[f"Recall@{k}"].mean()
        avg[f"NDCG@{k}"] = user_df[f"NDCG@{k}"].mean()
        avg[f"HitRate@{k}"] = user_df[f"HitRate@{k}"].mean()

    summary_df = pd.DataFrame([avg])
    summary_df.insert(0, "num_users", len(user_df))

    return summary_df, user_df


# ============================================================
# Experiment runners
# ============================================================

def build_candidate_cache(
    users: List[str],
    user_profiles: Dict[str, Dict[str, np.ndarray]],
    paper_embeddings_norm: np.ndarray,
    paper_map: pd.DataFrame,
    faiss_index,
    train_interacted: Dict[str, set],
    candidate_k: int,
) -> Dict[str, pd.DataFrame]:

    cache = {}

    for i, uid in enumerate(users, 1):
        cache[uid] = retrieve_by_profile_vector(
            user_id=uid,
            user_profiles=user_profiles,
            paper_embeddings_norm=paper_embeddings_norm,
            paper_map=paper_map,
            faiss_index=faiss_index,
            top_k=candidate_k,
            exclude_row_idxs=train_interacted.get(uid, set()),
        )

        if i % 100 == 0:
            print(f"[candidate] {i}/{len(users)} users")

    return cache


def run_method(
    method_name: str,
    users: List[str],
    candidate_cache: Dict[str, pd.DataFrame],
    model: TwoTower,
    user_to_idx: Dict[str, int],
    user_profiles: Dict[str, Dict[str, np.ndarray]],
    paper_embeddings: np.ndarray,
    device: torch.device,
    gt_by_user: Dict[str, set],
    ks: List[int],
    semantic_weight: float,
    two_tower_weight: float,
    lambda_val: float,
    current_date: str,
    final_k: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:

    recs_by_user = {}

    for i, uid in enumerate(users, 1):
        try:
            recs_by_user[uid] = recommend_for_user(
                user_id=uid,
                candidate_cache=candidate_cache,
                model=model,
                user_to_idx=user_to_idx,
                user_profiles=user_profiles,
                paper_embeddings=paper_embeddings,
                device=device,
                semantic_weight=semantic_weight,
                two_tower_weight=two_tower_weight,
                lambda_val=lambda_val,
                current_date=current_date,
                final_k=final_k,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"[WARN] skip user={uid}: {e}")

        if i % 100 == 0:
            print(f"[{method_name}] {i}/{len(users)} users")

    summary_df, user_df = evaluate_predictions(
        recs_by_user=recs_by_user,
        gt_by_user=gt_by_user,
        ks=ks,
    )

    summary_df.insert(0, "method", method_name)
    user_df.insert(0, "method", method_name)

    return summary_df, user_df, recs_by_user


def recency_stats(
    recs_by_user: Dict[str, pd.DataFrame],
    gt_by_user: Dict[str, set],
    current_date: str,
    top_k_for_date: int = 10,
) -> Dict[str, float]:

    cur = pd.Timestamp(current_date)
    all_top = []

    for uid, rec_df in recs_by_user.items():
        top = rec_df.head(top_k_for_date).copy()
        top["user_id"] = uid
        all_top.append(top)

    if not all_top:
        return {}

    top_df = pd.concat(all_top, ignore_index=True)
    top_df["update_date"] = pd.to_datetime(top_df["update_date"], errors="coerce")

    days_old = (cur - top_df["update_date"]).dt.days

    stats = {
        "avg_update_date": (
            str(top_df["update_date"].mean().date())
            if top_df["update_date"].notna().any()
            else ""
        ),
        "recent_1yr_ratio": float((days_old <= 365).mean()),
        "recent_2yr_ratio": float((days_old <= 730).mean()),
    }

    summary_df, _ = evaluate_predictions(
        recs_by_user=recs_by_user,
        gt_by_user=gt_by_user,
        ks=[10],
    )

    for col in summary_df.columns:
        if col != "num_users":
            stats[col] = float(summary_df.iloc[0][col])

    return stats


def run_query_fixed_personalization_analysis(
    queries: List[str],
    users: List[str],
    encoder,
    paper_embeddings_norm: np.ndarray,
    paper_map: pd.DataFrame,
    faiss_index,
    model: TwoTower,
    user_to_idx: Dict[str, int],
    user_profiles: Dict[str, Dict[str, np.ndarray]],
    paper_embeddings: np.ndarray,
    device: torch.device,
    candidate_k: int = 500,
    top_k: int = 10,
    batch_size: int = 512,
) -> pd.DataFrame:

    rows = []

    for query_text in queries:
        print(f"\n[personalization query] {query_text}")

        base_candidates = retrieve_by_query_text(
            query_text=query_text,
            encoder=encoder,
            paper_embeddings_norm=paper_embeddings_norm,
            paper_map=paper_map,
            faiss_index=faiss_index,
            top_k=candidate_k,
        )

        semantic_ranked = base_candidates.copy()
        semantic_ranked = semantic_ranked.sort_values(
            "semantic_score",
            ascending=False,
        ).reset_index(drop=True)

        semantic_top = semantic_ranked.head(top_k)

        semantic_recs_by_user = {
            uid: semantic_top.copy()
            for uid in users
        }

        personalized_recs_by_user = {}

        for i, uid in enumerate(users, 1):
            try:
                scored = score_two_tower(
                    model=model,
                    user_id=uid,
                    candidates_df=base_candidates,
                    user_to_idx=user_to_idx,
                    user_profiles=user_profiles,
                    paper_embeddings=paper_embeddings,
                    device=device,
                    batch_size=batch_size,
                )

                ranked = apply_late_fusion(
                    scored_df=scored,
                    semantic_weight=0.5,
                    two_tower_weight=0.5,
                )

                personalized_recs_by_user[uid] = ranked.head(top_k).copy()

            except Exception as e:
                print(f"[WARN] personalization skip user={uid}: {e}")

            if i % 50 == 0:
                print(f"[personalization] {i}/{len(users)} users")

        valid_users = [
            uid for uid in users
            if uid in personalized_recs_by_user
        ]

        if len(valid_users) < 2:
            continue

        def avg_pairwise_overlap(recs_by_user):
            vals = []

            for i in range(len(valid_users)):
                for j in range(i + 1, len(valid_users)):
                    u1 = valid_users[i]
                    u2 = valid_users[j]

                    a = set(
                        recs_by_user[u1]
                        .head(top_k)["row_idx"]
                        .astype(int)
                        .tolist()
                    )

                    b = set(
                        recs_by_user[u2]
                        .head(top_k)["row_idx"]
                        .astype(int)
                        .tolist()
                    )

                    vals.append(len(a & b) / top_k)

            return float(np.mean(vals)) if vals else np.nan

        def avg_profile_paper_cosine(recs_by_user):
            vals = []

            for uid in valid_users:
                profile_vec = user_profiles[uid]["liked"].astype("float32")
                profile_vec = profile_vec / (
                    np.linalg.norm(profile_vec) + 1e-8
                )

                rec_row_idxs = (
                    recs_by_user[uid]
                    .head(top_k)["row_idx"]
                    .astype(int)
                    .values
                )

                sims = paper_embeddings_norm[rec_row_idxs] @ profile_vec
                vals.extend(sims.tolist())

            return float(np.mean(vals)) if vals else np.nan

        rows.append({
            "query": query_text,
            "method": "Semantic only",
            "avg_user_overlap@10": avg_pairwise_overlap(semantic_recs_by_user),
            "avg_profile_paper_cosine@10": avg_profile_paper_cosine(semantic_recs_by_user),
        })

        rows.append({
            "query": query_text,
            "method": "Semantic + Two-Tower",
            "avg_user_overlap@10": avg_pairwise_overlap(personalized_recs_by_user),
            "avg_profile_paper_cosine@10": avg_profile_paper_cosine(personalized_recs_by_user),
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paper_map", default="data/paper_id_map.csv")
    parser.add_argument("--emb", default="data/paper_embeddings.npy")
    parser.add_argument("--faiss_index", default="data/papers_faiss.index")
    parser.add_argument("--train", default="two_tower/data/train_arxividx.csv")
    parser.add_argument("--test", default="two_tower/data/test_arxividx.csv")
    parser.add_argument("--ckpt", default="two_tower/checkpoints/best_two_tower_arxividx.pt")
    parser.add_argument("--out_dir", default="eval_outputs")

    parser.add_argument("--candidate_k", type=int, default=1000)
    parser.add_argument("--final_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_users", type=int, default=None)

    parser.add_argument("--min_train_liked", type=int, default=2)
    parser.add_argument("--min_test_liked", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--current_date", default="2026-05-25")

    parser.add_argument("--lambda_weak", type=float, default=0.01)
    parser.add_argument("--lambda_default", type=float, default=0.05)
    parser.add_argument("--lambda_strong", type=float, default=0.10)

    parser.add_argument("--run_llm", action="store_true")
    parser.add_argument("--llm_model", default="gemini-2.5-flash")
    parser.add_argument("--llm_top_candidates", type=int, default=50)

    parser.add_argument("--user_query", default="")
    parser.add_argument("--query_text", default="")
    parser.add_argument("--seed_title", default="")

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("[load] data")

    paper_map = safe_read_csv(args.paper_map)
    train_df = safe_read_csv(args.train)
    test_df = safe_read_csv(args.test)

    paper_embeddings = np.load(args.emb).astype("float32")
    paper_embeddings_norm = normalize_rows(paper_embeddings)

    paper_map["row_idx"] = paper_map["row_idx"].astype(int)
    paper_map["paper_id"] = paper_map["paper_id"].astype(str)

    emb_dim = paper_embeddings.shape[1]

    print("paper_map:", paper_map.shape)
    print("paper_embeddings:", paper_embeddings.shape)
    print("train:", train_df.shape)
    print("test:", test_df.shape)

    print("[load] checkpoint")

    ckpt = torch.load(args.ckpt, map_location=device)

    user_to_idx = {
        str(k): int(v)
        for k, v in ckpt["user_to_idx"].items()
    }

    config = ckpt["config"]

    if ckpt["emb_dim"] != emb_dim:
        raise ValueError(
            f"Embedding dim mismatch: ckpt={ckpt['emb_dim']}, emb={emb_dim}"
        )

    model = TwoTower(
        num_users=len(user_to_idx),
        emb_dim=ckpt["emb_dim"],
        user_id_dim=config["user_id_dim"],
        hidden_dim=config["hidden_dim"],
        out_dim=config["out_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[build] user profiles")

    user_profiles = build_user_profiles(
        train_df=train_df,
        paper_embeddings=paper_embeddings,
        emb_dim=emb_dim,
    )

    train_interacted = build_train_interacted_map(train_df)

    users = get_eligible_users(
        train_df=train_df,
        test_df=test_df,
        user_to_idx=user_to_idx,
        min_train_liked=args.min_train_liked,
        min_test_liked=args.min_test_liked,
        max_users=args.max_users,
        seed=args.seed,
    )

    if not users:
        raise ValueError(
            "No eligible users found. "
            "Try lowering --min_train_liked or --min_test_liked."
        )

    print("eligible users:", len(users))

    gt_by_user = {}

    for uid, g in test_df[test_df["label"] == 1].groupby("user_id"):
        gt_by_user[str(uid)] = set(g["row_idx"].astype(int).tolist())

    faiss_index = load_faiss_index(args.faiss_index)

    print("[build] candidate cache")

    candidate_cache = build_candidate_cache(
        users=users,
        user_profiles=user_profiles,
        paper_embeddings_norm=paper_embeddings_norm,
        paper_map=paper_map,
        faiss_index=faiss_index,
        train_interacted=train_interacted,
        candidate_k=args.candidate_k,
    )

    # ========================================================
    # 1. Offline Evaluation
    # ========================================================

    print("\n[run] offline evaluation")

    offline_summary, offline_user, recs_default = run_method(
        method_name="Semantic + Two-Tower + Recency",
        users=users,
        candidate_cache=candidate_cache,
        model=model,
        user_to_idx=user_to_idx,
        user_profiles=user_profiles,
        paper_embeddings=paper_embeddings,
        device=device,
        gt_by_user=gt_by_user,
        ks=[5, 10, 20],
        semantic_weight=0.5,
        two_tower_weight=0.5,
        lambda_val=args.lambda_default,
        current_date=args.current_date,
        final_k=args.final_k,
        batch_size=args.batch_size,
    )

    offline_summary.to_csv(out_dir / "offline_eval.csv", index=False)
    offline_user.to_csv(out_dir / "user_level_offline_eval.csv", index=False)

    print(offline_summary)

    # ========================================================
    # 2. Ablation Study
    # ========================================================

    print("\n[run] ablation study")

    methods = [
        ("Semantic only", 1.0, 0.0, 0.0),
        ("Two-Tower only", 0.0, 1.0, 0.0),
        ("Semantic + Two-Tower", 0.5, 0.5, 0.0),
        ("Semantic + Two-Tower + Recency", 0.5, 0.5, args.lambda_default),
    ]

    ablation_rows = []
    user_rows = []
    recs_by_method = {}

    for name, sw, tw, lam in methods:
        summary, user_df, recs = run_method(
            method_name=name,
            users=users,
            candidate_cache=candidate_cache,
            model=model,
            user_to_idx=user_to_idx,
            user_profiles=user_profiles,
            paper_embeddings=paper_embeddings,
            device=device,
            gt_by_user=gt_by_user,
            ks=[10],
            semantic_weight=sw,
            two_tower_weight=tw,
            lambda_val=lam,
            current_date=args.current_date,
            final_k=args.final_k,
            batch_size=args.batch_size,
        )

        ablation_rows.append(summary)
        user_rows.append(user_df)
        recs_by_method[name] = recs

        print(summary)

    if args.run_llm:
        print("\n[run] Full system with LLM reranking")

        full_summary, full_user_df, full_recs = run_full_system_with_llm(
            users=users,
            candidate_cache=candidate_cache,
            model=model,
            user_to_idx=user_to_idx,
            user_profiles=user_profiles,
            paper_embeddings=paper_embeddings,
            device=device,
            gt_by_user=gt_by_user,
            ks=[10],
            current_date=args.current_date,
            final_k=10,
            batch_size=args.batch_size,
            llm_model=args.llm_model,
            llm_top_candidates=args.llm_top_candidates,
            train_df=train_df,
            user_query=args.user_query,
            query_text=args.query_text,
            seed_title=args.seed_title,
        )

        ablation_rows.append(full_summary)
        user_rows.append(full_user_df)
        recs_by_method["Full system"] = full_recs

        print(full_summary)

    ablation_df = pd.concat(ablation_rows, ignore_index=True)
    user_ablation_df = pd.concat(user_rows, ignore_index=True)

    ablation_df.to_csv(out_dir / "ablation.csv", index=False)
    user_ablation_df.to_csv(out_dir / "user_level_ablation.csv", index=False)

    # ========================================================
    # 3. Recency Analysis
    # ========================================================

    print("\n[run] recency analysis")

    recency_settings = [
        ("No recency", 0.0),
        ("Weak recency", args.lambda_weak),
        ("Default recency", args.lambda_default),
        ("Strong recency", args.lambda_strong),
    ]

    recency_rows = []

    for name, lam in recency_settings:
        _, _, recs = run_method(
            method_name=name,
            users=users,
            candidate_cache=candidate_cache,
            model=model,
            user_to_idx=user_to_idx,
            user_profiles=user_profiles,
            paper_embeddings=paper_embeddings,
            device=device,
            gt_by_user=gt_by_user,
            ks=[10],
            semantic_weight=0.5,
            two_tower_weight=0.5,
            lambda_val=lam,
            current_date=args.current_date,
            final_k=args.final_k,
            batch_size=args.batch_size,
        )

        stats = recency_stats(
            recs_by_user=recs,
            gt_by_user=gt_by_user,
            current_date=args.current_date,
            top_k_for_date=10,
        )

        stats = {
            "setting": name,
            "lambda": lam,
            **stats,
        }

        recency_rows.append(stats)
        print(pd.DataFrame([stats]))

    recency_df = pd.DataFrame(recency_rows)
    recency_df.to_csv(out_dir / "recency_analysis.csv", index=False)

    # ========================================================
    # 4. Query-fixed Personalization Analysis
    # ========================================================

    print("\n[run] query-fixed personalization analysis")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("sentence-transformers/allenai-specter")

    personalization_queries = [
        "RAG evaluation for large language models",
        "self supervised learning for computer vision",
        "graph neural network recommender system",
    ]

    sample_users = users[: min(30, len(users))]

    personalization_df = run_query_fixed_personalization_analysis(
        queries=personalization_queries,
        users=sample_users,
        encoder=encoder,
        paper_embeddings_norm=paper_embeddings_norm,
        paper_map=paper_map,
        faiss_index=faiss_index,
        model=model,
        user_to_idx=user_to_idx,
        user_profiles=user_profiles,
        paper_embeddings=paper_embeddings,
        device=device,
        candidate_k=min(args.candidate_k, 500),
        top_k=10,
        batch_size=args.batch_size,
    )

    personalization_df.to_csv(
        out_dir / "personalization_analysis.csv",
        index=False,
    )

    print(personalization_df)

    print("\n[DONE] saved outputs to:", out_dir.resolve())

    for p in sorted(out_dir.glob("*.csv")):
        print("-", p)


if __name__ == "__main__":
    main()