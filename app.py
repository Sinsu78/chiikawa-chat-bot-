import base64
import html
import os
import re
import unicodedata
import warnings
from datetime import date
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch

# 코사인 유사도가 음수일 때 나오는 경고는 정상 동작이라 숨김
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

BASE_DIR = Path(__file__).parent
# chromadb(1.5.x)는 경로에 한글이 있으면 HNSW 인덱스를 못 읽음 → 앱 폴더로 이동 후 상대 경로 사용
os.chdir(BASE_DIR)
load_dotenv(BASE_DIR / ".env", override=True)

GPT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL")
EMBED_MODEL = os.getenv("EMBEDED_MODEL")
CHROMA_DIR = BASE_DIR / "chroma4"

KEEP_TAGS = ("NNG", "NNP", "SL", "SN", "XR")   # 명사, 외국어, 숫자, 어근만 사용 (조사·동사 제거)
# 거의 모든 질문·문서에 나와서 검색을 흐리는 단어
STOPWORDS = {"치이카와", "먼작귀", "에피소드", "장면", "내용", "애니", "캐릭터"}
# 입문자가 흔히 부르는 별칭 -> 공식 이름 (질문과 DB 문서 양쪽에 적용)
ALIASES = {
    "토끼": "우사기",
    "고양이": "하치와레",
    "다람쥐": "모몽가", "하늘다람쥐": "모몽가", "날다람쥐": "모몽가",
    "해달": "랏코",
    "밤만쥬": "쿠리만주",
}
ALIAS_TEXT = ", ".join(f"{k}={v}" for k, v in ALIASES.items())
RRF_K = 60       # RRF 상수 (보통 60)
FETCH_K = 20     # BM25, 벡터 검색에서 각각 가져올 후보 수
CHUNKS_PER_EPISODE = 2   # 에피소드 하나당 자료로 쓸 최대 청크 수
FIELD_LABELS = {"summary": "줄거리", "key_scene": "장면", "objects": "등장 사물"}

# 화면용 상수
# 제목 위에 보여 줄 이미지. 앞에서부터 처음 찾은 파일을 쓴다
HERO_IMAGES = [BASE_DIR / "assets" / name for name in ("hero.png", "hero.jpg", "hero.jpeg")]
AVATARS = {"user": "🙂", "assistant": "🐹"}
STARTER_QUESTIONS = [
    ("🐰", "우사기는 어떤 캐릭터야?"),
    ("🍜", "라면 먹는 에피소드 알려줘"),
    ("⚔️", "토벌이 뭐야?"),
    ("🐱", "하치와레랑 치이카와는 무슨 사이야?"),
]


# ---------------------------------------------------------------------------
# 1. 리소스 로딩 (앱이 재실행돼도 한 번만 만들도록 캐시)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_vectorstore():
    return Chroma(
        collection_name="chiikawa",                       # 적재할 때와 같은 이름
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_DIR.name,                # 상대 경로 "chroma4"
    )


@st.cache_resource
def load_kiwi():
    kiwi = Kiwi()
    # 캐릭터 이름이 '치이카+와'처럼 잘못 잘리지 않도록 사전에 등록
    for name in ("치이카와", "하치와레", "우사기", "모몽가", "쿠리만주", "랏코", "시사", "먼작귀"):
        kiwi.add_user_word(name, "NNP")
    return kiwi


def tokenize(text):
    """'치이카와가 라면 먹는 장면' -> ['라면'], '토끼가 나오는' -> ['우사기']"""
    tokens = [ALIASES.get(t.form, t.form) for t in load_kiwi().tokenize(text)
              if t.tag in KEEP_TAGS]
    return [t for t in tokens if t not in STOPWORDS]


@st.cache_resource
def load_bm25():
    """Chroma에 저장된 문서 전체로 BM25 인덱스를 만든다."""
    stored = load_vectorstore().get(include=["documents", "metadatas"])
    docs = [
        Document(id=i, page_content=text, metadata=meta)
        for i, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])
    ]
    return BM25Okapi([tokenize(d.page_content) for d in docs]), docs


@st.cache_resource
def load_llm():
    return init_chat_model(f"openai:{GPT_MODEL}")


@st.cache_resource
def load_web_search():
    return TavilySearch(max_results=3)


# ---------------------------------------------------------------------------
# 2. 프롬프트
# ---------------------------------------------------------------------------
class Grade(BaseModel):
    """2차 판정 결과. 입문자의 이름·전제 착각은 바로잡아서 DB로 답하게 한다."""
    related: bool          # 치이카와(작품·캐릭터·작가·굿즈 등)와 관련된 질문인지
    verdict: Literal["answerable", "correctable", "unanswerable"]
    correction: str = ""   # correctable일 때만: 질문과 자료가 어떻게 다른지 한 문장


# 치이카와와 무관한 질문에 대한 고정 답변 (LLM을 부르지 않음)
OFF_TOPIC_ANSWER = (
    "제가 가진 자료에는 없어요. 저는 치이카와(먼작귀)에 대한 질문만 답할 수 있어요. "
    "아래 질문처럼 물어봐 주세요!"
)
OFF_TOPIC_SUGGESTIONS = ["우사기는 어떤 캐릭터야?", "라면 먹는 에피소드 알려줘"]

# 치이카와 관련이지만 에피소드 DB에 없어 웹에서 답할 때 앞에 붙이는 고정 문구
WEB_NOTICE = ("ℹ️ 에피소드 자료에는 없어서 웹에서 찾아봤어요. "
              "웹 정보는 정확하지 않을 수 있으니 아래 출처도 함께 확인해 주세요.")

# 답변 끝에 LLM이 추천 질문을 적는 구분자. 화면에는 버튼으로 바꿔서 보여 준다
SUGGEST_MARKER = "[추천 질문]"
MAX_SUGGESTIONS = 2


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "치이카와 입문자의 [질문]을 두 가지로 판단해.\n\n"
     "1) related: 질문이 치이카와(먼작귀)의 캐릭터·에피소드·세계관·작가·굿즈·인기 등과 "
     "관련 있으면 true. 날씨, 뉴스, 공부, 다른 작품처럼 치이카와와 무관하면 false.\n\n"
     "2) verdict: 아래 [자료]로 질문에 답할 수 있는지.\n"
     "- answerable: 자료만으로 질문에 그대로 답할 수 있음\n"
     "- correctable: 질문의 핵심 대상(음식·물건·사건·장면)은 자료에 있지만, "
     "사용자가 등장인물이나 세부 내용을 잘못 알고 있음. "
     "correction에 무엇이 다른지 한 문장으로 적어 "
     "(예: '피자 만두를 먹는 건 치이카와가 아니라 우사기').\n"
     "- unanswerable: 질문의 핵심 대상 자체가 자료에 없음 "
     "(작가·제작 정보, 현실 정보, 자료에 없는 설정 등)\n"
     "입문자는 캐릭터 이름을 헷갈리는 게 자연스러워. "
     f"흔한 별칭: {ALIAS_TEXT}. 별칭만 다르고 내용이 맞으면 answerable이야. "
     "단, 핵심 대상이 자료에 없는데 비슷한 내용으로 억지로 끼워 맞추지 마.\n\n"
     "[자료]\n{context}"),
    ("human", "[질문] {question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 치이카와 안내자야. 오늘 날짜는 {today}야. "
     "아래 [자료]만 근거로 한국어로 답해. 자료에 없는 내용은 모른다고 말해. "
     "에피소드를 말할 때는 '4화 브로콜리, 피자만두'처럼 화수와 제목을 함께 적어. "
     "출처 링크는 화면에서 따로 붙이니 답변에는 출처를 적지 마. "
     f"질문에 별칭({ALIAS_TEXT})을 썼으면 공식 이름도 같이 알려줘.\n\n"
     "[말투 규칙]\n"
     "1. 치이카와를 처음 접하는 사람에게 설명하듯 항상 친절한 존댓말(~요)로 답해. "
     "토벌, 자격증처럼 작품 속 용어는 처음 나올 때 짧게 풀어서 설명해.\n"
     "2. 답변은 최소 2문장 이상으로 쓰고, 자료에 있는 구체적인 장면이나 예시를 곁들여.\n"
     f"3. 답변을 다 쓴 뒤 마지막에 '{SUGGEST_MARKER}' 한 줄을 쓰고, 그 아래에 "
     f"사용자가 이어서 물어볼 만한 질문 {MAX_SUGGESTIONS}개를 '- 질문' 형식으로 적어. "
     "질문은 사용자가 챗봇에게 묻는 말투로 짧게 써 (예: 우사기는 어떤 캐릭터야?)."
     "{note}\n\n"
     "[자료]\n{context}"),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# 3. 검색 로직 (노트북의 hybrid_search / search_web)
# ---------------------------------------------------------------------------
def format_docs(docs):
    """같은 에피소드의 청크를 한 블록으로 묶는다."""
    groups = {}  # episode_id -> [doc, ...]
    for d in docs:
        groups.setdefault(d.metadata.get("episode_id", ""), []).append(d)

    blocks = []
    for episode_id, ep_docs in groups.items():
        meta = ep_docs[0].metadata
        lines = [
            f"[{episode_id}] {meta.get('title', '')} (등장: {meta.get('characters', '')})",
            f"영상: {meta.get('video_url', '')}",
        ]
        for d in ep_docs:
            kind = FIELD_LABELS.get(d.metadata.get("field"), "")
            body = d.page_content.partition("\n")[2]   # 첫 줄 [제목 | 캐릭터] 접두어 제외
            lines.append(f"- ({kind}) {body}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def hybrid_search(question):
    """BM25 순위와 벡터 순위를 RRF(Reciprocal Rank Fusion)로 합친다. 청크 단위로 반환."""
    bm25, bm25_docs = load_bm25()
    scores = bm25.get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:FETCH_K]
    bm25_hits = [(bm25_docs[i], None) for i in top if scores[i] > 0]
    vector_hits = load_vectorstore().similarity_search_with_relevance_scores(question, k=FETCH_K)

    fused = {}  # doc_id -> {"doc", "rrf", "bm25", "vector", "vector_score"}
    for name, hits in (("bm25", bm25_hits), ("vector", vector_hits)):
        for rank, (d, score) in enumerate(hits, start=1):
            item = fused.setdefault(d.id, {"doc": d, "rrf": 0.0, "bm25": None,
                                           "vector": None, "vector_score": None})
            item["rrf"] += 1 / (RRF_K + rank)
            item[name] = rank
            if name == "vector":
                item["vector_score"] = score
    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)


def group_by_episode(items, k):
    """청크를 에피소드별로 묶어 상위 k개 에피소드만 남긴다.
    에피소드 순위는 가장 점수가 높은 청크 기준, 에피소드당 청크는 CHUNKS_PER_EPISODE개까지."""
    groups = {}  # episode_id -> [item, ...]  (items가 점수순이라 먼저 들어온 에피소드가 상위)
    for item in items:
        group = groups.setdefault(item["doc"].metadata.get("episode_id", ""), [])
        if len(group) < CHUNKS_PER_EPISODE:
            group.append(item)
    top_groups = list(groups.values())[:k]
    return [dict(item, episode_rank=rank)
            for rank, group in enumerate(top_groups, start=1) for item in group]


def passes_first_filter(item, threshold):
    """BM25에서 키워드가 맞았거나, 벡터 점수가 임계값 이상이면 통과."""
    if item["bm25"] is not None:
        return True
    return item["vector_score"] is not None and item["vector_score"] >= threshold


def search_chroma(question, threshold, k):
    """1차: 키워드/점수 필터, 2차: LLM 판정. 화면에 보여 주려고 중간 결과도 같이 반환."""
    results = group_by_episode(hybrid_search(question), k)

    # 1차 필터
    passed = [item for item in results if passes_first_filter(item, threshold)]
    docs = [item["doc"] for item in passed]

    # 2차 판정: 1차에서 다 탈락해도 '치이카와 관련 질문인지'는 판단해야 하므로 항상 호출
    grader = GRADE_PROMPT | load_llm().with_structured_output(Grade)
    grade = grader.invoke({
        "context": format_docs(docs) if docs else "(검색된 자료 없음)",
        "question": question,
    })

    base = {"scored": results, "related": grade.related, "correction": ""}
    if not docs:
        return {**base, "docs": [], "stage": "1차 필터에서 탈락"}
    if grade.verdict == "answerable":
        return {**base, "docs": docs, "stage": "2차 판정 통과"}
    if grade.verdict == "correctable":
        return {**base, "docs": docs, "stage": "2차 판정: 바로잡아 답변",
                "correction": grade.correction}
    return {**base, "docs": [], "stage": "2차 판정에서 탈락"}


def search_web(question):
    query = question if "치이카와" in question else f"치이카와 {question}"
    res = load_web_search().invoke({"query": query})
    items = res.get("results", []) if isinstance(res, dict) else []
    context = "\n\n".join(
        f"[웹] {r['title']} ({r['url']})\n{r['content']}" for r in items
    )
    return context, items


def retrieve(question, threshold, k):
    """답변 생성 전까지의 단계. 어디서 자료를 가져올지 결정한다."""
    db = search_chroma(question, threshold, k)
    if db["docs"]:
        # 출처 표시용: 자료로 쓴 에피소드 (순서 유지, 중복 제거)
        episodes = {}
        for d in db["docs"]:
            meta = d.metadata
            episodes.setdefault(meta["episode_id"], {
                "episode_id": meta["episode_id"],
                "title": meta.get("title", ""),
                "video_url": meta.get("video_url", ""),
            })
        return {
            "source": "DB",
            "context": format_docs(db["docs"]),
            "stage": db["stage"],
            "scored": db["scored"],
            "web": [],
            "episodes": list(episodes.values()),
            "correction": db["correction"],
        }

    # 치이카와와 무관한 질문은 웹 검색도 하지 않고 고정 답변
    if not db["related"]:
        return {
            "source": "없음",
            "context": "",
            "stage": db["stage"] + " · 치이카와와 무관한 질문",
            "scored": db["scored"],
            "web": [],
            "episodes": [],
            "correction": "",
        }

    context, items = search_web(question)
    return {
        "source": "웹",
        "context": context or "(검색 결과 없음)",
        "stage": db["stage"],
        "scored": db["scored"],
        "web": items,
        "episodes": [],
        "correction": "",
    }


def build_sources(answer, info):
    """답변 아래 붙일 출처 링크를 '칩' 모양 HTML로 만든다.
    제목·URL은 외부 데이터라 html.escape로 이스케이프한다 (XSS 방지).
    LLM 답변 본문은 HTML 허용 없이 따로 렌더링하므로 섞지 않는다."""
    if info["source"] == "DB":
        episodes = info["episodes"]
        # 답변에 'N화'로 언급된 에피소드만 출처로. 언급이 없으면 자료로 쓴 에피소드 전부
        mentioned = [ep for ep in episodes
                     if re.search(rf"(?<!\d){int(ep['episode_id'][2:])}화", answer)]
        pairs = [(ep["title"], ep["video_url"]) for ep in (mentioned or episodes)]
        icon, label = "📺", "출처 영상"
    else:
        pairs = [(r["title"], r["url"]) for r in info["web"]]
        icon, label = "🔗", "출처 웹페이지"
    if not pairs:
        return ""
    chips = "".join(
        f'<a class="src-chip" href="{html.escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{icon} {html.escape(title)}</a>'
        for title, url in pairs
    )
    return f'<div class="src-wrap"><div class="src-label">{label}</div>{chips}</div>'


def stream_answer(question, context, correction=""):
    note = ""
    if correction:
        note = (f"\n\n[바로잡을 점] {correction}\n"
                "사용자가 헷갈리는 건 자연스러운 일이니, 답변 처음에 이 점을 친절하게 "
                "바로잡은 뒤 자료로 답해.")
    chain = ANSWER_PROMPT | load_llm() | StrOutputParser()
    return chain.stream({
        "context": context,
        "question": question,
        "today": date.today().isoformat(),
        "note": note,
    })


# ---------------------------------------------------------------------------
# 4. 스타일 (CSS 주입)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css");

:root {
  --accent: #FF6F91;
  --accent-soft: #FF8FAB;
  --accent-warm: #FFB199;
  --ink: #2B2D42;
  --muted: #8A8FA3;
  --card: rgba(255, 255, 255, 0.75);
  --line: rgba(43, 45, 66, 0.08);
  --grad: linear-gradient(135deg, #FF8FAB 0%, #FFB199 100%);
}

/* 폰트: 아이콘 폰트(Material Symbols)는 건드리지 않도록 상속으로만 적용 */
.stApp { font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, sans-serif; }

/* 배경: 은은한 파스텔 메쉬 그라디언트 */
.stApp {
  background:
    radial-gradient(1100px 600px at 0% -10%, #FFE3EC 0%, transparent 60%),
    radial-gradient(900px 520px at 110% 0%, #E3F6EE 0%, transparent 55%),
    radial-gradient(700px 500px at 50% 120%, #FFF1DC 0%, transparent 60%),
    #FFFBF7;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 820px; padding-top: 2.2rem; padding-bottom: 8rem; }
footer { visibility: hidden; }

/* ---------- 히어로 ---------- */
.hero { text-align: center; margin: 0.5rem 0 2rem; }
.hero-img {
  display: block; margin: 0 auto 1.1rem;
  width: min(260px, 70%); height: auto;
  border-radius: 28px;
  box-shadow: 0 18px 40px -22px rgba(255, 111, 145, .6);
  animation: floaty 4s ease-in-out infinite;
}
@keyframes floaty { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@media (prefers-reduced-motion: reduce) { .hero-img { animation: none; } }
.hero-badge {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .35rem .9rem; border-radius: 999px;
  background: rgba(255, 143, 171, .12); color: #E0557A;
  border: 1px solid rgba(255, 143, 171, .3);
  font-size: .8rem; font-weight: 600;
}
.hero-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: #3DD68C;
  box-shadow: 0 0 0 3px rgba(61, 214, 140, .2); }
.hero .hero-title {
  font-size: 3.6rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.15;
  margin: .8rem 0 .4rem;
}
.hero .hero-text {
  background: linear-gradient(135deg, #FF6F91 0%, #FF9671 55%, #FFC75F 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero .hero-sub { color: var(--muted); font-size: 1rem; }

/* ---------- 채팅 말풍선 ---------- */
[data-testid="stChatMessage"] {
  background: var(--card);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 1.1rem 1.25rem;
  margin-bottom: .9rem;
  box-shadow: 0 10px 30px -18px rgba(43, 45, 66, .25);
  animation: fadeUp .35s ease both;
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* 사용자 말풍선은 그라디언트로 강조 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"][aria-label*="user"] {
  background: var(--grad);
  border: none;
  margin-left: 12%;
  box-shadow: 0 12px 28px -14px rgba(255, 111, 145, .55);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"][aria-label*="user"] p { color: #fff; font-weight: 500; }

[data-testid="stChatMessage"] p { line-height: 1.75; }
[data-testid="stChatMessage"] strong { color: #E0557A; }

/* ---------- 출처 칩 ---------- */
.src-wrap { display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; margin: .4rem 0 .2rem; }
.src-label { width: 100%; font-size: .75rem; font-weight: 700; color: var(--muted);
  letter-spacing: .04em; text-transform: uppercase; }
.src-chip {
  display: inline-flex; align-items: center; gap: .3rem;
  padding: .38rem .8rem; border-radius: 999px;
  background: #fff; border: 1px solid var(--line);
  color: var(--ink) !important; text-decoration: none !important;
  font-size: .85rem; font-weight: 500;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.src-chip:hover { transform: translateY(-1px); border-color: rgba(255, 111, 145, .45);
  box-shadow: 0 6px 16px -8px rgba(255, 111, 145, .5); }

/* ---------- 버튼 (추천 질문 칩) ---------- */
.stButton > button {
  border-radius: 999px;
  border: 1px solid rgba(255, 143, 171, .4);
  background: #fff; color: #E0557A;
  font-weight: 600; padding: .45rem 1rem;
  box-shadow: 0 3px 10px -6px rgba(255, 111, 145, .5);
  transition: all .18s ease;
}
.stButton > button:hover {
  background: var(--grad); color: #fff; border-color: transparent;
  transform: translateY(-1px);
}
.stButton > button:focus:not(:active) { border-color: var(--accent); color: #E0557A; }
.stButton > button:hover:focus { color: #fff; }

/* 첫 화면 시작 질문 카드 */
[class*="st-key-starter"] button {
  border-radius: 18px; width: 100%; min-height: 76px;
  justify-content: flex-start; text-align: left;
  padding: .9rem 1.1rem; color: var(--ink);
  border: 1px solid var(--line); background: var(--card);
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 24px -18px rgba(43, 45, 66, .35);
}
[class*="st-key-starter"] button p { font-size: .95rem; font-weight: 600; }

.caption-hint { color: var(--muted); font-size: .82rem; font-weight: 600; margin: .3rem 0 .4rem; }

/* ---------- 디버그 expander ---------- */
[data-testid="stExpander"] details {
  border-radius: 14px; border: 1px dashed rgba(43, 45, 66, .16);
  background: rgba(250, 250, 252, .7);
}
[data-testid="stExpander"] summary p { font-size: .82rem; color: var(--muted); }

/* ---------- 입력창 ---------- */
[data-testid="stBottom"] > div { background: linear-gradient(to top, #FFFBF7 65%, rgba(255, 251, 247, 0)); }
[data-testid="stChatInput"] > div {
  border-radius: 26px !important;
  border: 1.5px solid rgba(255, 143, 171, .35) !important;
  background: #fff !important;
  box-shadow: 0 14px 34px -16px rgba(255, 111, 145, .45);
}
[data-testid="stChatInput"]:focus-within > div { border-color: var(--accent) !important; }
[data-testid="stChatInputSubmitButton"] {
  background: var(--grad) !important; border-radius: 50% !important; color: #fff !important;
}

/* ---------- 사이드바 ---------- */
[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, .62);
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h2 { font-weight: 800; letter-spacing: -0.02em; }
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #FFF0F4 0%, #FFF7EE 100%);
  border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.1rem;
}
[data-testid="stMetricValue"] { font-weight: 800; color: #E0557A; }
[data-testid="stSidebar"] .stButton > button {
  border-radius: 14px; color: var(--ink); border-color: var(--line); box-shadow: none;
}
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_hero_image():
    """assets/ 폴더의 히어로 이미지를 data URI로 읽는다. 파일이 없으면 None."""
    for path in HERO_IMAGES:
        if path.exists():
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode()
            return f"data:{mime};base64,{data}"
    return None


def render_hero():
    src = load_hero_image()
    img = f'<img class="hero-img" src="{src}" alt="">' if src else ""
    st.markdown(
        f"""
        <div class="hero">
          {img}
          <div class="hero-title"><span class="hero-text">우나야하bot</span> 🐹</div>
          <div class="hero-sub">에피소드 DB에서 먼저 찾고, 없으면 웹에서 찾아 답해요.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 5. 화면
# ---------------------------------------------------------------------------
def _is_punct(ch):
    return unicodedata.category(ch)[0] in "PS"   # 문장부호(P), 기호(S)


def fix_bold(text):
    """'**나가노(ナガノ)**예요'처럼 닫는 ** 앞이 문장부호이고 뒤에 조사가 붙으면
    마크다운(CommonMark)이 굵게로 인식하지 못한다. 앞뒤 문장부호를 ** 바깥으로 뺀다.
    예) '**나가노(ナガノ)**예요' -> '**나가노(ナガノ**)예요'"""
    def repl(m):
        inner = m.group(1)
        start, end = 0, len(inner)
        while start < end and _is_punct(inner[start]):
            start += 1
        while end > start and _is_punct(inner[end - 1]):
            end -= 1
        if start == end:
            return m.group(0)
        return f"{inner[:start]}**{inner[start:end]}**{inner[end:]}"
    return re.sub(r"\*\*(.+?)\*\*", repl, text)


def split_suggestions(text):
    """LLM 답변을 (본문, 추천 질문 목록)으로 나눈다."""
    body, _, tail = text.partition(SUGGEST_MARKER)
    # 스트리밍 도중 '[추천' 처럼 구분자 일부만 도착한 경우 본문에서 잘라 냄
    for i in range(len(SUGGEST_MARKER) - 1, 0, -1):
        if body.endswith(SUGGEST_MARKER[:i]):
            body = body[:-i]
            break
    suggestions = []
    for line in tail.splitlines():
        line = line.strip().lstrip("-•*0123456789. ").strip().strip("\"'")
        if line:
            suggestions.append(line)
    return body.strip(), suggestions[:MAX_SUGGESTIONS]


def write_answer_stream(stream):
    """st.write_stream 대신 직접 스트리밍하면서 매번 fix_bold를 적용한다.
    추천 질문 부분은 화면에 글자로 보이지 않게 빼고, (본문, 추천 질문)을 반환한다."""
    placeholder = st.empty()
    text = ""
    for chunk in stream:
        text += chunk
        placeholder.markdown(fix_bold(split_suggestions(text)[0]) + " ▍")   # 타이핑 커서
    body, suggestions = split_suggestions(text)
    body = fix_bold(body)
    placeholder.markdown(body)
    return body, suggestions


def ask_suggestion(text):
    """추천 질문 버튼 콜백: 다음 실행에서 이 질문을 입력한 것처럼 처리한다."""
    st.session_state.pending_question = text


def render_suggestions(suggestions, key_prefix):
    if not suggestions:
        return
    st.markdown('<div class="caption-hint">✨ 이어서 물어보세요</div>', unsafe_allow_html=True)
    cols = st.columns(len(suggestions))
    for i, (col, s) in enumerate(zip(cols, suggestions)):
        with col:
            st.button(s, key=f"{key_prefix}_{i}", on_click=ask_suggestion, args=(s,),
                      use_container_width=True)


def render_starters():
    """대화가 비어 있을 때 보여 주는 시작 질문 카드 (2x2)."""
    rows = [STARTER_QUESTIONS[i:i + 2] for i in range(0, len(STARTER_QUESTIONS), 2)]
    for r, row in enumerate(rows):
        cols = st.columns(2)
        for c, (col, (emoji, q)) in enumerate(zip(cols, row)):
            with col:
                st.button(f"{emoji}  {q}", key=f"starter_{r}_{c}",
                          on_click=ask_suggestion, args=(q,), use_container_width=True)


def render_debug(info):
    """검색 과정(점수, 판정, 웹 결과)을 접이식으로 보여 준다."""
    with st.expander(f"🔍 검색 과정 · 출처: {info['source']} · {info['stage']}"):
        st.markdown("**하이브리드 검색 결과 (상위 k개 에피소드)**")
        st.caption(f"BM25 검색어: {', '.join(tokenize(info['question'])) or '(없음)'}")
        if info.get("correction"):
            st.caption(f"2차 판정이 찾은 착각: {info['correction']}")
        st.dataframe(
            [
                {
                    "에피소드 순위": item["episode_rank"],
                    "RRF": round(item["rrf"], 4),
                    "BM25 순위": item["bm25"],
                    "벡터 순위": item["vector"],
                    "벡터 점수": None if item["vector_score"] is None
                                 else round(item["vector_score"], 3),
                    "통과": "O" if passes_first_filter(item, info["threshold"]) else "X",
                    "문서 ID": item["doc"].id,
                    "내용": item["doc"].page_content[:60],
                }
                for item in info["scored"]
            ],
            hide_index=True,
            use_container_width=True,
        )
        if info["web"]:
            st.markdown("**웹 검색 결과**")
            for r in info["web"]:
                st.markdown(f"- [{r['title']}]({r['url']})")


st.set_page_config(page_title="치이카와 안내자", page_icon="🐹", layout="centered")
inject_css()
render_hero()

if not CHROMA_DIR.exists():
    st.error(f"Chroma DB 폴더가 없어요: {CHROMA_DIR}\n노트북에서 먼저 적재해 주세요.")
    st.stop()

with st.sidebar:
    st.header("⚙️ 검색 설정")
    threshold = st.slider("1차 필터 임계값", 0.0, 1.0, 0.3, 0.05,
                          help="BM25 키워드가 맞았거나, 코사인 유사도가 이 값 이상인 문서만 "
                               "2차 판정으로 보냅니다.")
    k = st.slider("검색 에피소드 수 (k)", 1, 10, 5,
                  help=f"에피소드 단위로 묶어서 가져옵니다. 에피소드당 청크는 최대 {CHUNKS_PER_EPISODE}개.")
    show_debug = st.toggle("검색 과정 보기", value=True)
    st.divider()
    st.metric("📚 DB 문서 수", f"{load_vectorstore()._collection.count():,}")
    if st.button("🗑️ 대화 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 입력창 또는 추천 질문 버튼으로 들어온 질문 (입력창은 화면 맨 아래에 고정됨)
question = (st.chat_input("치이카와에 대해 무엇이든 물어보세요 ✨")
            or st.session_state.pop("pending_question", None))

messages = st.session_state.messages

# 첫 화면: 대화가 없으면 시작 질문 카드
if not messages and not question:
    render_starters()

# 지난 대화 다시 그리기 (Streamlit은 입력마다 스크립트 전체를 재실행함)
for idx, msg in enumerate(messages):
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg.get("sources_html"):
            st.markdown(msg["sources_html"], unsafe_allow_html=True)
        if show_debug and msg.get("info"):
            render_debug(msg["info"])
        # 추천 질문 버튼은 마지막 답변에만, 새 질문이 들어온 실행에서는 숨김
        if idx == len(messages) - 1 and not question:
            render_suggestions(msg.get("suggestions", []), key_prefix=f"sugg_{idx}")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        suggestions = []
        sources_html = ""
        try:
            with st.spinner("자료 찾는 중..."):
                info = retrieve(question, threshold, k)
                info["threshold"] = threshold
                info["question"] = question
            if info["source"] == "없음":
                answer = OFF_TOPIC_ANSWER
                suggestions = OFF_TOPIC_SUGGESTIONS
                st.markdown(answer)
            else:
                if info["source"] == "웹":
                    st.info(WEB_NOTICE)
                answer, suggestions = write_answer_stream(
                    stream_answer(question, info["context"], info["correction"]))
                if info["source"] == "웹":
                    answer = WEB_NOTICE + "\n\n" + answer   # 대화 기록에도 남도록
            sources_html = build_sources(answer, info)
            if sources_html:
                st.markdown(sources_html, unsafe_allow_html=True)
        except Exception as e:
            info = None
            answer = f"오류가 발생했어요: {e}"
            st.error(answer)

        if show_debug and info:
            render_debug(info)
        render_suggestions(suggestions, key_prefix=f"sugg_{len(messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "info": info,
         "suggestions": suggestions, "sources_html": sources_html}
    )