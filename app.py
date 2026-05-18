import streamlit as st
import pandas as pd
import numpy as np
import os
import json

# ── 페이지 기본 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="PaperLens · 논문 추천 시스템",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (깔끔한 화이트 & 모던 스타일) ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* 전체 배경 화이트 톤 & 기본 폰트 */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f8f9fa;
    color: #212529;
}
.stApp { background-color: #f8f9fa; }

/* 사이드바 (차분한 라이트 그레이) */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e9ecef;
}
section[data-testid="stSidebar"] * { color: #495057 !important; }

/* 헤더 */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0.2rem;
    font-weight: bold;
}
.hero-sub {
    color: #6c757d;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* 모드 카드 */
.mode-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.7rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.mode-card:hover { 
    border-color: #3b82f6; 
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59,130,246,0.08);
}
.mode-card.active {
    border-color: #1e3a8a;
    background: #f0f4ff;
}
.mode-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 20px;
    background: #e0e7ff;
    color: #1e3a8a;
    margin-bottom: 0.4rem;
}
.mode-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1f2937;
}
.mode-desc {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 4px;
}

/* 결과 카드 */
.result-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    position: relative;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.01);
    transition: all 0.2s;
}
.result-card:hover { 
    border-color: #3b82f6; 
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.04);
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1e3a8a, #3b82f6);
}
.rank-badge {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #3b82f622;
    position: absolute;
    right: 1.5rem;
    top: 1.2rem;
    font-weight: bold;
}
.paper-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.45;
    margin-bottom: 0.6rem;
    padding-right: 3rem;
}
.paper-meta {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
}
.meta-chip {
    font-size: 0.72rem;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 3px 12px;
    color: #4b5563;
    font-weight: 500;
}
.meta-chip.gold {
    border-color: #bfdbfe;
    color: #1e3a8a;
    background: #eff6ff;
}
.reason-box {
    background: #f9fafb;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.6;
}
.reason-label {
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #2563eb;
    font-weight: 600;
    margin-bottom: 4px;
}

/* 입력 필드 */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #1f2937 !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px #3b82f622 !important;
}

/* 실행 버튼 */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 6px rgba(30,58,138,0.15) !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.95 !important;
    transform: translateY(-1px) !important;
}

/* [개선] 구린 기본 디자인의 로딩 스피너 및 사이드바 에러 스타일 커스텀 오버라이드 */
div[data-testid="stSidebar"] [data-testid="stAlert"], 
div[data-testid="stSidebar"] .stSpinner {
    background-color: #f0f4ff !important;
    border: 1px solid #bfdbfe !important;
    color: #1e3a8a !important;
    border-radius: 10px !important;
}
div[data-testid="stSidebar"] .stSpinner > div {
    border-top-color: #1e3a8a !important;
}

/* 안내 배너 */
.info-banner {
    background: #f0f4ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    font-size: 0.85rem;
    color: #1e3a8a;
    margin-bottom: 1.2rem;
}
.info-banner b { color: #1d4ed8; font-weight: 600; }

.section-label {
    font-size: 0.75rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #1e3a8a;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

hr.gold { border-color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 & 모델 로드 (캐시) ──────────────────────────────────
@st.cache_resource(show_spinner="시스템 자원 및 모델 아티팩트 로딩 중...")
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
    
    # 최신 google-genai 클라이언트 초기화
    genai.configure(api_key=gemini_key)
    client = None

    # [수정] 가장 바깥쪽 data/ 폴더 내부를 정확히 지리하도록 경로 세팅 보완
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
        client_llm=client, DEVICE=DEVICE,
    )


# ── 추천 알고리즘 파이프라인 ────────────────────────────────────
def run_recommend(sys, user_id, input_type, user_query, seed_title, seed_abstract,
                  query_text, candidate_k, final_k, alpha, beta,
                  semantic_weight, two_tower_weight, lambda_val):
    import torch
    from google.genai import types

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

    # 1. Candidate Retrieval (FAISS)
    if input_type == "seed":
        cands = search(encode(build_text(seed_title, seed_abstract)), candidate_k)
    elif input_type == "query":
        cands = search(encode(query_text), candidate_k)
    else:  # seed_query 혼합 탐색
        sv = encode(build_text(seed_title, seed_abstract))
        qv = encode(query_text)
        fv = alpha * qv + beta * sv
        fv = (fv / (np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8)).astype("float32")
        cands = search(fv, candidate_k)

    # 2. Deep Scoring (Two-Tower Model Inference)
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

    # 3. Late Fusion Ranking
    def mnorm(x):
        x = np.asarray(x, dtype=np.float32)
        mn, mx = x.min(), x.max()
        return np.ones_like(x) * 0.5 if abs(mx - mn) < 1e-8 else (x - mn) / (mx - mn)
    cands["sem_n"] = mnorm(cands["semantic_score"].values)
    cands["tt_n"]  = mnorm(cands["two_tower_score"].values)
    t = semantic_weight + two_tower_weight
    cands["final_score"] = (semantic_weight/t) * cands["sem_n"] + (two_tower_weight/t) * cands["tt_n"]
    cands = cands.sort_values("final_score", ascending=False).reset_index(drop=True)

    # 4. Recency-aware Time Decay
    cands["update_date"] = pd.to_datetime(cands["update_date"], errors="coerce")
    cur = pd.Timestamp.now()
    cands["months_passed"] = ((cur.year - cands["update_date"].dt.year) * 12 + (cur.month - cands["update_date"].dt.month)).clip(lower=0)
    cands["time_penalty"] = np.exp(-lambda_val * cands["months_passed"])
    cands["time_adjusted_score"] = cands["final_score"] * cands["time_penalty"]
    cands = cands.sort_values("time_adjusted_score", ascending=False).reset_index(drop=True)

    # 5. LLM User Agent Personalized Reranking (Gemini 3 Flash Image 표준 모델인 최신 플래시 탑재)
    client = sys["client_llm"]
    papers_info = ""
    for _, row in cands.head(20).iterrows():
        papers_info += f"[Paper ID: {row['paper_id']}]\nTitle: {row['title']}\nUpdate Date: {row['update_date']}\n---\n"

    prompt = f"""You are an AI Research Assistant. The user wants papers about: "{user_query or query_text or seed_title}"

Here is a list of top candidate papers ranked by our internal system:
{papers_info}

Select the Top {final_k} papers that best match the user's intent.
Provide a brief 1-sentence reason WHY each paper is recommended.

Output STRICTLY as a valid JSON array:
[{{"paper_id": "1234.5678", "reason": "..."}}]
"""
    try:
        # 최신 google-genai 라이브러리 인터페이스 규격 매핑
        resp = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                temperature=0.3
            )
        )
        reranked = json.loads(resp.text)
        results = []
        for rank, item in enumerate(reranked):
            pid, reason = item.get("paper_id"), item.get("reason")
            matched = cands[cands["paper_id"] == pid]
            if not matched.empty:
                r = matched.iloc[0]
                results.append({
                    "final_rank": rank + 1, "paper_id": pid,
                    "title": r["title"], "reason": reason,
                    "update_date": r["update_date"],
                    "categories": r.get("categories", ""),
                    "time_adjusted_score": r["time_adjusted_score"],
                })
        
        # 완벽한 붕괴 방지용 가드 클로즈 결합
        if not results:
            fallback_df = cands.head(final_k).copy()
            fallback_df["final_rank"] = range(1, len(fallback_df) + 1)
            fallback_df["reason"] = "LLM 정렬 매칭 실패로 순수 통합 랭킹 점수 결과만 유지합니다."
            return fallback_df
            
        return pd.DataFrame(results).head(final_k)
        
    except Exception as e:
        st.warning(f"LLM 구조화 파싱/호출 오류: {e}. 시스템 백오프 기반 기존 스코어로 대체합니다.")
        fallback_df = cands.head(final_k).copy()
        fallback_df["final_rank"] = range(1, len(fallback_df) + 1)
        fallback_df["reason"] = "API 게이트웨이 지연 오류로 인해 하이브리드 통합 점수 상위권을 대체 반환합니다."
        return fallback_df


# ══════════════════════════════════════════════
#  UI 레이아웃 구현 및 렌더링
# ══════════════════════════════════════════════

# ── 상단 헤더 ──────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem;">
  <div class="hero-title">🔭 PaperLens</div>
  <div class="hero-sub">AI-Powered Academic Paper Recommendation Engine</div>
</div>
<hr class="gold">
""", unsafe_allow_html=True)

# ── 사이드바 관제탑 패널 ─────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">⚙ 시스템 제어판</div>', unsafe_allow_html=True)

    try:
        sys = load_system()
        sys_loaded = True
    except Exception as e:
        st.error(f"오브젝트 로드 실패: {e}")
        sys_loaded = False

    if sys_loaded:
        st.markdown("**초개인화 대상 유저 필터링**")
        user_options = sys["user_list"][:200]  # 다이나믹 렌더링 부하 절감을 위한 Top 200 슬라이싱
        selected_user = st.selectbox(
            "유저 가상 프로필 ID",
            options=user_options,
            format_func=lambda x: f"User Agent profile [{x}]",
            help="학습 데이터셋 로그에 연동된 실제 익명화 유저 아티팩트 목록입니다."
        )

        st.markdown('<hr class="gold">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🎚 가중치 하이퍼파라미터</div>', unsafe_allow_html=True)

        lambda_val = st.slider(
            "시간 감쇠 계수 (Time Decay λ)",
            min_value=0.0, max_value=0.3, value=0.05, step=0.01,
            help="상승할수록 과거 논문에 강한 패널티를 부여해 신규 논문 위주로 랭킹을 방출합니다."
        )
        semantic_weight = st.slider(
            "프롬프트 의미 유사도 비중 (Semantic)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        )
        two_tower_weight = st.sidebar.slider(
            "임베딩 협업 필터링 비중 (Two-Tower)",
            min_value=0.0, max_value=1.0, value=1.0 - semantic_weight, step=0.05,
            disabled=True, help="의미 유사도 비율에 연동되어 100% 비율로 강제 오프셋 연동됩니다."
        )
        
        candidate_k = st.number_input("1차 후보 추출 풀 크기 (Recall K)", min_value=20, max_value=200, value=100, step=10)
        final_k     = st.number_input("최종 추천 타겟 수 (Top K)", min_value=3, max_value=20, value=10, step=1)

        st.markdown('<hr class="gold">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.75rem; color:#6b7280; line-height:1.8;">
        📦 인덱싱 논문 모음: <b style="color:#e8c97e">{sys['paper_map'].shape[0]:,}</b><br>
        👥 식별 유저 히스토리: <b style="color:#e8c97e">{len(sys['user_list']):,}</b><br>
        🖥 하드웨어 백엔드: <b style="color:#e8c97e">{str(sys['DEVICE']).upper()}</b>
        </div>
        """, unsafe_allow_html=True)


# ── 메인 워크스페이스 대시보드 ─────────────────────
if not sys_loaded:
    st.error("사이드바 리소스 연동 상태에 결함이 감지되었습니다. 에러 로그를 검토하세요.")
    st.stop()

# 정교한 인터랙티브 모드 탭 체계 구성
st.markdown('<div class="section-label">📋 탐색 모드 아키텍처 선택</div>', unsafe_allow_html=True)

MODE_OPTIONS = {
    "query": {
        "label": "MODE 1",
        "title": "자연어 프롬프트 검색",
        "desc": "추상적이거나 구체적인 요구 명세를 입력합니다",
    },
    "seed": {
        "label": "MODE 2",
        "title": "기준 논문 임베딩 확장",
        "desc": "특정 타겟 논문의 메타데이터 구조를 역추적합니다",
    },
    "seed_query": {
        "label": "MODE 3",
        "title": "하이브리드 다중 결합 조건",
        "desc": "기준 논문 조건 상태에서 가이드 쿼리로 초점을 조좁힙니다",
    },
}

cols = st.columns(3)
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "query"

for col, (mode_key, meta) in zip(cols, MODE_OPTIONS.items()):
    with col:
        active = "active" if st.session_state.input_mode == mode_key else ""
        st.markdown(f"""
        <div class="mode-card {active}" id="card_{mode_key}">
          <div class="mode-badge">{meta['label']}</div>
          <div class="mode-title">{meta['title']}</div>
          <div class="mode-desc">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"{meta['title']} 활성화", key=f"btn_{mode_key}", use_container_width=True):
            st.session_state.input_mode = mode_key
            st.rerun()

st.markdown('<hr class="gold">', unsafe_allow_html=True)

# ── 동적 가변 입력 컨테이너 ───────────────────────
mode = st.session_state.input_mode
st.markdown(f'<div class="section-label">✏ 가변 입력 파라미터 — {MODE_OPTIONS[mode]["title"]}</div>', unsafe_allow_html=True)

user_query   = ""
seed_title   = None
seed_abstract= None
query_text   = None
alpha, beta  = 0.7, 0.3
valid        = False

if mode == "query":
    st.markdown('<div class="info-banner">💬 <b>인텐트 탐색</b>: 연구 타겟 인공지능 도메인 패러다임을 입력하세요. (e.g., "Retrieval-Augmented Generation evaluation strategy")</div>', unsafe_allow_html=True)
    query_text = st.text_input("🔍 입력 프롬프트 수집기", placeholder="e.g. self-supervised learning on multimodal video framework")
    user_query = query_text
    valid = bool(query_text and query_text.strip())

elif mode == "seed":
    st.markdown('<div class="info-banner">📄 <b>기준 아티팩트</b>: 소유 중이거나 타겟팅 중인 연구 자료의 제목과 초록 벡터를 공급해 주세요.</div>', unsafe_allow_html=True)
    seed_title    = st.text_input("📄 시드 아카이브 타이틀", placeholder="e.g. Linformer: Self-Attention with Linear Complexity")
    seed_abstract = st.text_area("📝 시드 아카이브 초록 (Abstract)", height=120, placeholder="텍스트 데이터 원문을 주입하세요...")
    user_query = seed_title
    valid = bool(seed_title and seed_title.strip())

else:  # seed_query 결합 모드
    st.markdown('<div class="info-banner">🔥 <b>크로스오버 조율</b>: 시드 논문의 추상 경향성과 유저의 직관적인 가이드라인을 백분율 비율로 하이브리드 연산합니다.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        seed_title = st.text_input("📄 시드 아카이브 타이틀", placeholder="e.g. RETHINKING ATTENTION WITH PERFORMERS")
    with c2:
        query_text = st.text_input("🔍 유저 가이드라인 쿼리", placeholder="e.g. low-rank matrix approximation")
    seed_abstract = st.text_area("📝 시드 아카이브 초록 (선택 옵션)", height=100, placeholder="공란으로 둘 경우 타이틀 중심으로 가중 연산 벡터가 매핑됩니다.")
    ac1, ac2 = st.columns(2)
    with ac1:
        alpha = st.slider("α (유저 프롬프트 제어 가중치)", 0.0, 1.0, 0.7, 0.05)
    with ac2:
        beta  = st.slider("β (시드 아키텍처 상속 가중치)", 0.0, 1.0, 0.3, 0.05)
    user_query = query_text or seed_title
    valid = bool(seed_title and seed_title.strip() and query_text and query_text.strip())

# ── 오케스트레이션 실행 파트 ───────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("🚀  초개인화 추천 파이프라인 트리거", disabled=not valid)

if not valid:
    st.markdown('<p style="color:#6b7280; font-size:0.82rem; font-style:italic;">⚠️ 검색 활성화를 유도하기 위해 필수 메타데이터 입력을 완료해 주세요.</p>', unsafe_allow_html=True)

# ── 결과 피드 레이어 출력 ─────────────────────────
if run_btn and valid:
    with st.spinner("🔭 벡터 스페이스 스캔 및 생성형 유저 에이전트 다차원 재정렬 중..."):
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
            st.error(f"파이프라인 실행 중 치명적 예외 발생: {e}")

if "last_result" in st.session_state:
    result_df = st.session_state["last_result"]
    st.markdown('<hr class="gold">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">📑 최적 매칭 결과 피드 — {len(result_df)} 세트 스캔 완료</div>', unsafe_allow_html=True)

    for idx, row in result_df.iterrows():
        rank     = int(row.get("final_rank", idx + 1))
        title    = row.get("title", "Missing Title Description")
        reason   = row.get("reason", "")
        date_val = row.get("update_date", "")
        cats     = row.get("categories", "")
        score    = row.get("time_adjusted_score", 0.0)

        # 타임스탬프 유연성 정제 포맷팅
        try:
            date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")
        except:
            date_str = str(date_val)[:10] if pd.notna(date_val) else "Unknown Date"

        # 카테고리 태그 칩 빌더 (결측치 방어 코드 수립)
        cat_chips = ""
        if pd.notna(cats) and str(cats).strip():
            for c in str(cats).split()[:3]:
                cat_chips += f'<span class="meta-chip">{c}</span>'

        st.markdown(f"""
        <div class="result-card">
          <div class="rank-badge">#{rank}</div>
          <div class="paper-title">{title}</div>
          <div class="paper-meta">
            <span class="meta-chip gold">Final Score {score:.4f}</span>
            <span class="meta-chip">📅 {date_str}</span>
            {cat_chips}
          </div>
          {"<div class='reason-label'>🔬 AI 에이전트 페르소나의 추천 근거</div><div class='reason-box'>" + reason + "</div>" if reason else ""}
        </div>
        """, unsafe_allow_html=True)

    # 연구용 자산 아카이브 변환 오프라인 전송 (CSV)
    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥  분석 결과 학술 레포트 엑셀 데이터(CSV) 다운로드",
        data=csv_data,
        file_name="paperlens_academic_report.csv",
        mime="text/csv",
    )