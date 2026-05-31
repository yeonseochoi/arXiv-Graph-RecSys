import streamlit as st
import pandas as pd
import numpy as np
import os
import json

# ── 페이지 기본 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="PaperLens · AI 논문 추천 시스템",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 프리미엄 라이트 테마 CSS (Pretendard 기반) ───────────────────────
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

/* 전체 배경 및 폰트 세팅 */
html, body, [class*="css"] {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    background-color: #fcfcfd;
    color: #2d3748;
}
.stApp { background-color: #fcfcfd; }

/* 사이드바 디자인 (미니멀 화이트) */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #f1f5f9;
    box-shadow: 2px 0 12px rgba(0,0,0,0.01);
}
section[data-testid="stSidebar"] * { color: #475569 !important; }

/* 대시보드 메인 타이틀 */
.hero-title {
    font-family: 'Pretendard', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -1px;
    color: #1e3a8a;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
}

.mode-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 0.5rem;
    
    min-height: 170px;
    
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.01), 0 2px 4px -1px rgba(0,0,0,0.01);
}
.mode-card:hover { 
    border-color: #3b82f6; 
    transform: translateY(-3px);
    box-shadow: 0 12px 20px -3px rgba(59,130,246,0.08);
}
.mode-card.active {
    border-color: #2563eb;
    background: #f8fafc;
    box-shadow: 0 10px 15px -3px rgba(37,99,235,0.06);
}
.mode-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 30px;
    background: #eff6ff;
    color: #2563eb;
    margin-bottom: 0.5rem;
}
.mode-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1e293b;
}
.mode-desc {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 5px;
    line-height: 1.4;
    
    min-height: 55px; 
}

/* 추천 결과 카드 (가장 예뻐야 하는 곳) */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    transition: all 0.2s ease;
}
.result-card:hover { 
    border-color: #cbd5e1; 
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.04);
}
.rank-badge {
    font-size: 1.6rem;
    font-weight: 800;
    color: #2563eb;
    opacity: 0.15;
    position: absolute;
    right: 1.6rem;
    top: 1.2rem;
}
.paper-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.5;
    margin-bottom: 0.6rem;
    padding-right: 3rem;
}
.paper-title a {
    color: #0f172a !important;
    text-decoration: none !important;
    transition: color 0.2s ease;
}
.paper-title a:hover {
    color: #2563eb !important; 
}
.paper-meta {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.9rem;
}
.meta-chip {
    font-size: 0.75rem;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 30px;
    padding: 4px 12px;
    color: #475569;
    font-weight: 500;
}
.meta-chip.gold {
    border-color: #93c5fd;
    color: #1d4ed8;
    background: #eff6ff;
    font-weight: 600;
}
.reason-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 4px 12px 12px 4px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.6;
}
.reason-label {
    font-size: 0.75rem;
    color: #2563eb;
    font-weight: 700;
    margin-bottom: 4px;
}

/* 입력 위젯 스타일 */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
    font-size: 0.92rem !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* 메인 실행 버튼 */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.2) !important;
    border: none !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.3) !important;
}

/* 은은하고 세련된 사이드바 로딩창 매핑 */
div[data-testid="stSidebar"] [data-testid="stAlert"], 
div[data-testid="stSidebar"] .stSpinner {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.01) !important;
}
div[data-testid="stSidebar"] .stSpinner > div {
    border-top-color: #2563eb !important;
}

/* 안내판 배너 */
.info-banner {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #166534;
    margin-bottom: 1.5rem;
}
.info-banner b { color: #15803d; font-weight: 700; }

.section-label {
    font-size: 0.78rem;
    letter-spacing: 1px;
    color: #475569;
    font-weight: 700;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
}

hr.gold { border-color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 & 모델 로드 (캐시) ──────────────────────────────────
@st.cache_resource(show_spinner="시스템 리소스 및 추천 모델을 불러오는 중...")
def load_system():
    import faiss
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    import google.generativeai as genai
    from dotenv import load_dotenv

    load_dotenv()
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise EnvironmentError(".env 파일에 GEMINI_API_KEY가 세팅되어 있지 않습니다.")
    
    genai.configure(api_key=gemini_key)

    PAPER_MAP_PATH   = "data/paper_id_map.csv"
    PAPER_EMB_PATH   = "data/paper_embeddings.npy"
    FAISS_INDEX_PATH = "data/papers_faiss.index"
    TWO_TOWER_CKPT   = "two_tower/checkpoints/best_two_tower_arxividx.pt"
    TRAIN_PATH       = "two_tower/data/train_arxividx.csv"
    DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paper_map = pd.read_csv(PAPER_MAP_PATH)
    paper_map["paper_id"] = paper_map["paper_id"].astype(str)
    paper_embeddings = np.load(PAPER_EMB_PATH).astype("float32")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    encoder = SentenceTransformer("sentence-transformers/allenai-specter")

    class TwoTower(nn.Module):
        def __init__(self, num_users, emb_dim, user_id_dim=64, hidden_dim=256, out_dim=128):
            super().__init__()
            self.user_id_emb = nn.Embedding(num_users, user_id_dim)
            self.user_tower = nn.Sequential(
                nn.Linear(user_id_dim + emb_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim, out_dim),
            )
            self.item_tower = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim, out_dim),
            )
        def forward(self, user_idx, liked_vec, disliked_vec, item_vec):
            uid_vec = self.user_id_emb(user_idx)
            user_z = self.user_tower(torch.cat([uid_vec, liked_vec, disliked_vec], dim=1))
            item_z = self.item_tower(item_vec)
            user_z = nn.functional.normalize(user_z, dim=1)
            item_z = nn.functional.normalize(item_z, dim=1)
            return (user_z * item_z).sum(dim=1)

    ckpt = torch.load(TWO_TOWER_CKPT, map_location=DEVICE)
    user_to_idx = ckpt["user_to_idx"]
    config = ckpt["config"]
    emb_dim = ckpt["emb_dim"]

    model = TwoTower(
        num_users=len(user_to_idx), emb_dim=emb_dim,
        user_id_dim=config["user_id_dim"], hidden_dim=config["hidden_dim"], out_dim=config["out_dim"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_df = pd.read_csv(TRAIN_PATH, dtype={"user_id": str}, low_memory=False)
    train_df["user_id"] = train_df["user_id"].astype(str)

    user_profiles = {}
    for uid, g in train_df.groupby("user_id"):
        liked_idx    = g.loc[g["label"] == 1, "row_idx"].astype(int).values
        disliked_idx = g.loc[g["label"] == 0, "row_idx"].astype(int).values
        user_profiles[str(uid)] = {
            "liked":    paper_embeddings[liked_idx].mean(axis=0)    if len(liked_idx)    > 0 else np.zeros(emb_dim, dtype=np.float32),
            "disliked": paper_embeddings[disliked_idx].mean(axis=0) if len(disliked_idx) > 0 else np.zeros(emb_dim, dtype=np.float32),
        }

    user_list = train_df["user_id"].value_counts().index.tolist()
    return dict(
        paper_map=paper_map, paper_embeddings=paper_embeddings,
        faiss_index=faiss_index, encoder=encoder,
        two_tower=model, user_to_idx=user_to_idx, emb_dim=emb_dim,
        user_profiles=user_profiles, user_list=user_list,
        DEVICE=DEVICE,
    )


# ── 추천 알고리즘 파이프라인 ────────────────────────────────────
def run_recommend(sys, user_id, input_type, user_query, seed_title, seed_abstract,
                  query_text, candidate_k, final_k, alpha, beta,
                  semantic_weight, two_tower_weight, lambda_val):
    import torch
    import google.generativeai as genai

    def build_text(t, a):
        t = "" if pd.isna(t) else str(t).strip()
        a = "" if pd.isna(a) else str(a).strip()
        return (t + " " + a).strip()

    def encode(text):
        return sys["encoder"].encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def search(qvec, top_k, excl=None):
        scores, idxs = sys["faiss_index"].search(qvec, top_k + 10)
        res = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0: continue
            row = sys["paper_map"].iloc[i]
            pid = str(row["paper_id"])
            if excl and pid == str(excl): continue
            res.append({"row_idx": int(row["row_idx"]), "paper_id": pid,
                        "title": row["title"], "update_date": row["update_date"],
                        "categories": row["categories"], "semantic_score": float(s)})
            if len(res) == top_k: break
        return pd.DataFrame(res)

    # 1. 후보군 추출 (FAISS)
    if input_type == "seed":
        cands = search(encode(build_text(seed_title, seed_abstract)), candidate_k)
    elif input_type == "query":
        cands = search(encode(query_text), candidate_k)
    else:  
        sv = encode(build_text(seed_title, seed_abstract))
        qv = encode(query_text)
        fv = alpha * qv + beta * sv
        fv = (fv / (np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8)).astype("float32")
        cands = search(fv, candidate_k)

    # 2. 모델 기반 취향 스코어링 (Two-Tower 모델 순전파)
    uid_str = str(user_id)
    uid_idx = sys["user_to_idx"][uid_str]
    profile = sys["user_profiles"][uid_str]
    lv, dv = profile["liked"], profile["disliked"]
    item_vecs = sys["paper_embeddings"][cands["row_idx"].astype(int).values].astype("float32")
    all_scores = []
    DEVICE = sys["DEVICE"]
    with torch.no_grad():
        for s in range(0, len(item_vecs), 512):
            e = s + 512
            bi = item_vecs[s:e]; bs = len(bi)
            scores = sys["two_tower"](
                torch.full((bs,), uid_idx, dtype=torch.long, device=DEVICE),
                torch.tensor(np.repeat(lv[None], bs, 0), dtype=torch.float32, device=DEVICE),
                torch.tensor(np.repeat(dv[None], bs, 0), dtype=torch.float32, device=DEVICE),
                torch.tensor(bi, dtype=torch.float32, device=DEVICE),
            )
            all_scores.extend(scores.cpu().numpy().tolist())
    cands["two_tower_score"] = all_scores

    # 3. 랭킹 통합 (Late Fusion)
    def mnorm(x):
        x = np.asarray(x, dtype=np.float32)
        mn, mx = x.min(), x.max()
        return np.ones_like(x) * 0.5 if abs(mx - mn) < 1e-8 else (x - mn) / (mx - mn)
    cands["sem_n"] = mnorm(cands["semantic_score"].values)
    cands["tt_n"]  = mnorm(cands["two_tower_score"].values)
    t = semantic_weight + two_tower_weight
    cands["final_score"] = (semantic_weight/t) * cands["sem_n"] + (two_tower_weight/t) * cands["tt_n"]
    cands = cands.sort_values("final_score", ascending=False).reset_index(drop=True)

    # 4. 시간 경과 감쇠 (최신성 반영)
    cands["update_date"] = pd.to_datetime(cands["update_date"], errors="coerce")
    cur = pd.Timestamp.now()
    cands["months_passed"] = ((cur.year - cands["update_date"].dt.year) * 12 + (cur.month - cands["update_date"].dt.month)).clip(lower=0)
    cands["time_penalty"] = np.exp(-lambda_val * cands["months_passed"])
    cands["time_adjusted_score"] = cands["final_score"] * cands["time_penalty"]
    cands = cands.sort_values("time_adjusted_score", ascending=False).reset_index(drop=True)

    # 5. 생성형 LLM 재정렬 및 추천 사유 생성 (Gemini 2.5 Flash 고정)
    model_llm = genai.GenerativeModel("gemini-2.5-flash")
    papers_info = ""
    for rank, (_, row) in enumerate(cands.head(50).iterrows(), start=1):
        papers_info += (
            f"[Internal Rank: {rank}]\n"
            f"Paper ID: {row['paper_id']}\n"
            f"Title: {row['title']}\n"
            f"Categories: {row.get('categories', '')}\n"
            f"Update Date: {row['update_date']}\n"
            f"Recency-adjusted Score: {row['time_adjusted_score']:.6f}\n"
            "---\n"
        )

    prompt = f"""
    You are an AI Research Assistant. The user wants papers about:
    "{user_query or query_text or seed_title}"

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
                temperature=0.3
            )
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
                "time_adjusted_score": r["time_adjusted_score"],
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
                "time_adjusted_score": r["time_adjusted_score"],
            })
            selected_ids.add(pid)

            if len(results) == final_k:
                break

        return pd.DataFrame(results)
        
    except Exception as e:
        st.warning(f"AI 재정렬 프로세스 일시 지연: {e}")
        fallback_df = cands.head(final_k).copy()
        fallback_df["final_rank"] = range(1, len(fallback_df) + 1)
        fallback_df["reason"] = "네트워크 지연으로 인해 시스템 알고리즘 추천 스코어 기반으로 제공합니다."
        return fallback_df


# ══════════════════════════════════════════════
#  UI 레이아웃 구현 및 렌더링
# ══════════════════════════════════════════════

# ── 대시보드 상단 헤더 ─────────────────────────────
st.markdown("""
<div style="padding: 1.2rem 0 0.8rem;">
  <div class="hero-title">🔭 PaperLens</div>
  <div class="hero-sub">나만을 위한 초개인화 학술 논문 추천 시스템</div>
</div>
<hr class="gold">
""", unsafe_allow_html=True)

# ── 사이드바 제어판 (사용자 친화적 단어로 전면 수정) ──────────────────
with st.sidebar:
    st.markdown('<div class="section-label">👤 추천 대상 설정</div>', unsafe_allow_html=True)

    try:
        sys = load_system()
        sys_loaded = True
    except Exception as e:
        st.error(f"데이터 연결 실패: {e}")
        sys_loaded = False

    if sys_loaded:
        user_options = sys["user_list"][:200]  
        selected_user = st.selectbox(
            "유저 ID 선택",
            options=user_options,
            format_func=lambda x: f"시뮬레이션 유저 {x}",
            help="선택한 유저가 과거에 읽었던 논문 기록과 취향을 기반으로 맞춤 추천을 시작합니다."
        )

        st.markdown('<hr class="gold">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">⚙️ 추천 기준 정밀 설정</div>', unsafe_allow_html=True)

        lambda_val = st.slider(
            "🕒 최신 연구 우선 반영도",
            min_value=0.0, max_value=0.3, value=0.01, step=0.01,
            help="오른쪽으로 밀수록 최근에 발표된 트렌디한 논문을 상단에 먼저 배치하고, 왼쪽으로 밀수록 발간 시기와 상관없이 고전 논문들을 함께 보여줍니다."
        )

        semantic_weight = st.slider(
            "🎯 현재 검색어 집중도 (vs 과거 취향)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="100%에 가까울수록 내가 방금 입력한 검색 키워드 위주로 논문을 찾고, 0%에 가까울수록 검색어보다는 유저가 과거에 좋아했던 논문 성향을 중심으로 매칭합니다."
        )

        two_tower_weight = st.sidebar.slider(
            "가중치 자동 분배",
            min_value=0.0, max_value=1.0, value=1.0 - semantic_weight, step=0.05,
            disabled=True, 
            help="위의 '현재 검색어 집중도'를 조절하면, 전체 비중(100%) 중 남은 비율이 유저의 과거 취향 반영 비중으로 자동 계산됩니다."
        )
        
        candidate_k = st.number_input("1차 검색 후보 수", min_value=20, max_value=200, value=100, step=10, help="알고리즘이 전체 논문 중 1차적으로 뽑아낼 후보의 개수입니다.")
        final_k     = st.number_input("최종 화면 노출 개수", min_value=3, max_value=20, value=10, step=1)

        st.markdown('<hr class="gold">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.75rem; color:#64748b; line-height:1.8;">
        📦 전체 보유 논문: <b style="color:#1e3a8a">{sys['paper_map'].shape[0]:,} 편</b><br>
        👤 등록된 유저 수: <b style="color:#1e3a8a">{len(sys['user_list']):,} 명</b><br>
        🖥 연산 디바이스: <b style="color:#1e3a8a">{str(sys['DEVICE']).upper()}</b>
        </div>
        """, unsafe_allow_html=True)


# ── 메인 바디 워크스페이스 ─────────────────────
if not sys_loaded:
    st.stop()

# 세련된 모드 카드 라벨 세팅
st.markdown('<div class="section-label">📋 어떤 방법으로 논문을 찾으시겠어요?</div>', unsafe_allow_html=True)

MODE_OPTIONS = {
    "query": {
        "label": "방법 1",
        "title": "💡 키워드로 찾기",
        "desc": "관심 있는 연구 주제나 키워드를 입력해 관련 논문을 탐색합니다.",
    },
    "seed": {
        "label": "방법 2",
        "title": "📄 기준 논문으로 찾기",
        "desc": "알고 있는 논문의 제목과 초록을 기반으로 가장 비슷한 논문을 확장 탐색합니다.",
    },
    "seed_query": {
        "label": "방법 3",
        "title": "🔥 기준 논문 + 키워드 혼합",
        "desc": "특정 논문을 기준으로 삼은 뒤, 원하는 키워드를 더해 정밀하게 검색합니다.",
    },
}

cols = st.columns(3)
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "query"

for col, (mode_key, meta) in zip(cols, MODE_OPTIONS.items()):
    with col:
        active = "active" if st.session_state.input_mode == mode_key else ""
        st.markdown(f"""
        <div class="mode-card {active}">
          <div class="mode-badge">{meta['label']}</div>
          <div class="mode-title">{meta['title']}</div>
          <div class="mode-desc">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"이 방법 선택", key=f"btn_{mode_key}", use_container_width=True):
            st.session_state.input_mode = mode_key
            st.rerun()

st.markdown('<hr class="gold">', unsafe_allow_html=True)

# ── 가변 입력 폼 컨테이너 ───────────────────────
mode = st.session_state.input_mode
st.markdown(f'<div class="section-label">✏ 정보 입력 — {MODE_OPTIONS[mode]["title"]}</div>', unsafe_allow_html=True)

user_query   = ""
seed_title   = None
seed_abstract= None
query_text   = None
alpha, beta  = 0.7, 0.3
valid        = False

if mode == "query":
    st.markdown('<div class="info-banner">💡 <b>자연어 키워드 모드</b>: 찾고 싶은 인공지능/컴퓨터 공학 연구 분야를 영문으로 입력해 주세요. (예: "RAG evaluation matrix")</div>', unsafe_allow_html=True)
    query_text = st.text_input("🔍 검색할 내용을 적어주세요", placeholder="e.g. computer vision self supervised learning transformer")
    user_query = query_text
    valid = bool(query_text and query_text.strip())

elif mode == "seed":
    st.markdown('<div class="info-banner">📄 <b>기준 논문 확장 모드</b>: 인상 깊게 읽었거나 연구의 기준이 되는 논문의 영문 제목과 초록을 입력하세요.</div>', unsafe_allow_html=True)
    seed_title    = st.text_input("📄 기준 논문 제목 입력", placeholder="e.g. Attention Is All You Need")
    seed_abstract = st.text_area("📝 기준 논문 초록(Abstract) 원문 붙여넣기", height=120, placeholder="논문 사이트에서 초록을 복사해 붙여넣어 주세요...")
    user_query = seed_title
    valid = bool(seed_title and seed_title.strip())

else: 
    st.markdown('<div class="info-banner">🔥 <b>복합 모드</b>: 기준 논문의 기본 맥락을 따라가면서, 유저가 원하는 키워드의 비중(%)을 조절해 검색을 수행합니다.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        seed_title = st.text_input("📄 기준 논문 제목", placeholder="e.g. RETHINKING ATTENTION WITH PERFORMERS")
    with c2:
        query_text = st.text_input("🔍 좁혀 들어갈 추가 검색어", placeholder="e.g. efficient attention mechanism")
    seed_abstract = st.text_area("📝 기준 논문 초록 (선택 사항)", height=100, placeholder="초록을 적어주시면 논문 매칭이 훨씬 정교해집니다.")
    ac1, ac2 = st.columns(2)
    with ac1:
        alpha = st.slider(
            "🔍 새 검색어에 더 집중하기", 
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="이 값을 높이면 기준 논문보다는 방금 입력창에 입력한 '추가 검색어'에 해당되는 새로운 논문들을 집중적으로 찾아옵니다."
        )
    with ac2:
        beta  = st.slider(
            "📄 기존 논문의 주제 유지하기", 
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            help="이 값을 높이면 새로운 검색어를 입력했더라도, 원래 기준 논문이 다루던 핵심 학술 분야와 맥락을 벗어나지 않도록 합니다."
        )
    user_query = query_text or seed_title
    valid = bool(seed_title and seed_title.strip() and query_text and query_text.strip())

# ── 추천 작동 엔진 작동 패널 ───────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("🔭  나에게 맞는 맞춤 논문 추천받기", disabled=not valid)

# ── 결과 피드 렌더링 카드 ─────────────────────────
if run_btn and valid:
    with st.spinner("🔭 분석 엔진을 가동하여 가장 알맞은 논문을 정렬하고 있습니다..."):
        try:
            result_df = run_recommend(
                sys=sys,
                user_id=selected_user,
                input_type=mode,
                user_query=user_query,
                seed_title=seed_title,
                seed_abstract=seed_abstract,
                query_text=query_text,
                candidate_k=int(candidate_k),
                final_k=int(final_k),
                alpha=alpha,
                beta=beta,
                semantic_weight=semantic_weight,
                two_tower_weight=1.0 - semantic_weight,
                lambda_val=lambda_val,
            )
            st.session_state["last_result"] = result_df
        except Exception as e:
            st.error(f"추천 시스템 연산 중 예상치 못한 에러가 포착되었습니다: {e}")

if "last_result" in st.session_state:
    result_df = st.session_state["last_result"]
    st.markdown('<hr class="gold">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">📑 🎉 검색 완료! 유저 맞춤형 논문 리스트 {len(result_df)}편</div>', unsafe_allow_html=True)

    for idx, row in result_df.iterrows():
        rank     = int(row.get("final_rank", idx + 1))
        
        paper_id = str(row.get("paper_id", ""))
        
        title    = row.get("title", "제목이 없는 학술지 양식입니다.")
        reason   = row.get("reason", "")
        date_val = row.get("update_date", "")
        cats     = row.get("categories", "")
        score    = row.get("time_adjusted_score", 0.0)

        try:
            date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")
        except:
            date_str = str(date_val)[:10] if pd.notna(date_val) else "날짜 정보 없음"

        cat_chips = ""
        if pd.notna(cats) and str(cats).strip():
            for c in str(cats).split()[:3]:
                cat_chips += f'<span class="meta-chip">{c}</span>'

        # 카드 렌더링
        st.markdown(f"""
        <div class="result-card">
          <div class="rank-badge">#{rank}</div>
          <div class="paper-title">
            <a href="https://arxiv.org/abs/{paper_id}" target="_blank">{title}</a>
          </div>
          <div class="paper-meta">
            <span class="meta-chip gold">종합 매칭 점수 {score:.4f}</span>
            <span class="meta-chip">📅 발행일: {date_str}</span>
            {cat_chips}
          </div>
          {"<div class='reason-label'>💡 AI가 이 논문을 추천하는 이유</div><div class='reason-box'>" + reason + "</div>" if reason else ""}
        </div>
        """, unsafe_allow_html=True)

    # 엑셀 보고서 추출
    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥  추천받은 논문 리스트 엑셀 데이터(CSV) 다운로드",
        data=csv_data,
        file_name="paperlens_recommend_report.csv",
        mime="text/csv",
    )