# @title  
#Ollama
import os
import re
import zipfile
import pickle
import hashlib
import json
import requests

import numpy as np
import faiss
import streamlit as st
import gdown
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer, CrossEncoder


# ==========================================================
# CONFIG
# ==========================================================
FILE_ID       = "1zrJOzLjnOIBuVVbTW0FsX38V6xIlpjV2"
ZIP_PATH      = "ncert.zip"
EXTRACT_DIR   = "ncert"

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 80
BATCH_SIZE    = 64

EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL      = "llama3.2:3b"
OLLAMA_URL        = "http://localhost:11434"

RETRIEVE_K   = 30
RERANK_TOP_K = 5

INDEX_PATH = "faiss_index.bin"
META_PATH  = "chunks_meta.pkl"
HASH_PATH  = "index_hash.txt"

# ==========================================================
# STREAMLIT PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="NCERT Flashcard Generator", layout="wide")
st.title("📘 NCERT → Smart Concept Flashcard + Active Learning")

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.header("⚙️ Index Controls")
    st.caption(f"Chunk: `{CHUNK_SIZE}` | Overlap: `{CHUNK_OVERLAP}` | Retrieve K: `{RETRIEVE_K}`")
    st.caption("Chunking: **hybrid paragraph-overlap**")
    if st.button("🔄 Force Rebuild Index"):
        for p in [INDEX_PATH, META_PATH, HASH_PATH]:
            if os.path.exists(p):
                os.remove(p)
        st.cache_resource.clear()
        st.success("Index cleared — reloading...")
        st.rerun()
    st.divider()
    st.header("ℹ️ Models")
    st.caption(f"Embed: `{EMBED_MODEL_NAME}`")
    st.caption(f"Reranker: `{RERANK_MODEL_NAME}`")
    st.caption(f"Summarizer: `{OLLAMA_MODEL}` via Ollama")
    if OLLAMA_URL:
        st.caption(f"Ollama: `{OLLAMA_URL}`")
    else:
        st.warning("⚠️ Set OLLAMA_URL in config!")
    if os.path.exists(INDEX_PATH):
        st.metric("Index size", f"{os.path.getsize(INDEX_PATH)/1e6:.1f} MB")

# ==========================================================
# NOISE PATTERNS
# ==========================================================
NOISE_RE = re.compile(
    r'Ch_\d+\.indd\s*\d+[^\n]*'
    r'|\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}'
    r'|Prelims\.indd.*'
    r'|ISBN[^\n]*'
    r'|Reprint[^\n]*'
    r'|Printed[^\n]*'
    r'|All rights reserved[^\n]*'
    r'|Copyright[^\n]*'
    r'|\d{1,2}\s[A-Za-z]+\s\d{4}',
    re.IGNORECASE
)

# ==========================================================
# SECTION TYPE DETECTION
# ==========================================================
def detect_section_type(text):
    t = text.lower()
    if any(k in t for k in [
        "pedagogical hints", "weblinks", "project idea",
        "after reading this chapter", "contents", "reprint",
        "students should be made to understand",
        "teachers can ask",
    ]):
        return "noise"
    if any(k in t for k in [
        "exercise", "very short answer", "answer the following",
        "fill in the blank", "true or false", "questions for practice",
        "long answer type", "short answer type", "mcq", "multiple choice"
    ]):
        return "exercise"
    if any(k in t for k in [
        "is defined as", "is called", "refers to", "may be defined",
        "can be defined", "definition", "means the", "means that",
        "dimensions of", "consists of", "includes economic",
        "constituting the", "environment means", "is the sum total",
        "sum total of", "totality of", "can be described as",
        "is characterised", "features of", "characteristics of",
        "meaning of", "types of", "kinds of",
    ]):
        return "definition"
    if any(k in t for k in [
        "summary", "key points", "in this chapter",
        "we have learnt", "let us recapitulate", "to summarise",
        "in brief", "in conclusion"
    ]):
        return "summary"
    if any(k in t for k in [
        "for example", "for instance", "e.g.",
        "case study", "let us understand", "illustration"
    ]):
        return "example"
    if any(k in t for k in [
        "activity", "project", "do it yourself",
        "think and discuss", "intext question"
    ]):
        return "activity"
    return "theory"

# ==========================================================
# RECURSIVE ZIP EXTRACTOR
# ==========================================================
def extract_all_zips(folder):
    found_new = True
    while found_new:
        found_new = False
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".zip"):
                    zip_path   = os.path.join(root, file)
                    extract_to = os.path.join(root, os.path.splitext(file)[0])
                    if not os.path.exists(extract_to):
                        try:
                            with zipfile.ZipFile(zip_path, "r") as zf:
                                zf.extractall(extract_to)
                            found_new = True
                        except Exception as e:
                            st.warning(f"Could not extract {file}: {e}")

# ==========================================================
# DOWNLOAD + EXTRACT
# ==========================================================
@st.cache_resource
def download_and_extract(file_id):
    if not os.path.exists(ZIP_PATH):
        with st.spinner("Downloading NCERT dataset..."):
            success = False
            try:
                gdown.download(id=file_id, output=ZIP_PATH, quiet=False, fuzzy=True)
                success = os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 10000
            except Exception as e:
                st.warning(f"Primary download failed: {e}. Trying fallback...")
            if not success:
                try:
                    session  = requests.Session()
                    URL      = "https://drive.google.com/uc?export=download"
                    response = session.get(URL, params={"id": file_id}, stream=True)
                    token    = next(
                        (v for k, v in response.cookies.items()
                         if k.startswith("download_warning")), None
                    )
                    if token:
                        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
                    else:
                        response = session.get(
                            f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t",
                            stream=True
                        )
                    with open(ZIP_PATH, "wb") as f:
                        for chunk in response.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)
                    if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 10000:
                        raise ValueError("Downloaded file too small.")
                except Exception as e2:
                    st.error(f"Both download methods failed: {e2}")
                    st.stop()

    if not zipfile.is_zipfile(ZIP_PATH):
        os.remove(ZIP_PATH)
        st.error("Downloaded file is not a valid ZIP.")
        st.stop()

    if not os.path.exists(EXTRACT_DIR):
        with st.spinner("Extracting outer ZIP..."):
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(EXTRACT_DIR)

    with st.spinner("Extracting subject ZIPs..."):
        extract_all_zips(EXTRACT_DIR)

    pdf_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(EXTRACT_DIR)
        for f in files if f.lower().endswith(".pdf")
    ]
    st.info(f"📄 PDFs found: {len(pdf_files)}")
    return EXTRACT_DIR, pdf_files

# ==========================================================
# PDF READING
# ==========================================================
def read_pdf(path):
    try:
        doc        = fitz.open(path)
        all_text   = ""
        for page in doc:
            blocks = page.get_text("blocks", sort=False)
            blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
            if not blocks:
                continue
            page_width = page.rect.width
            mid_x      = page_width / 2
            left_blocks  = [b for b in blocks if b[0] < mid_x - 20]
            right_blocks = [b for b in blocks if b[0] >= mid_x - 20]
            if left_blocks and right_blocks and len(right_blocks) >= 2:
                ordered = (
                    sorted(left_blocks,  key=lambda b: b[1]) +
                    sorted(right_blocks, key=lambda b: b[1])
                )
            else:
                ordered = sorted(blocks, key=lambda b: (round(b[1] / 15) * 15, b[0]))
            page_text = "\n\n".join(b[4].strip() for b in ordered if b[4].strip())
            all_text += page_text + "\n\n"
        doc.close()
        return all_text
    except Exception:
        return ""

# ==========================================================
# EXTRACT HEADINGS FROM PDF USING FONT SIZE
# ==========================================================
def clean_heading(h):
    # fix missing spaces: "Meaningof" → "Meaning of", "Howa" → "How a"
    # insert space before capital letters that follow lowercase
    h = re.sub(r'([a-z])([A-Z])', r'\1 \2', h)
    # fix "wordsas" "valuesas" — lowercase+lowercase run-ons before common words
    h = re.sub(r'([a-z])(as|of|in|and|the|to|by|for|on)\b', r'\1 \2', h)
    h = re.sub(r'\b(as|of|in|and|the|to|by|for|on)([A-Z])', r'\1 \2', h)

    # remove spaced letters: "M eaning" → "Meaning"
    h = re.sub(r'\b([A-Z])\s([a-z])', r'\1\2', h)

    # remove leading bullets
    h = re.sub(r'^\([ivxlcdm\s]+\)\s*', '', h, flags=re.IGNORECASE)
    h = re.sub(r'^\d+[\.\)]\s*', '', h)
    h = re.sub(r'^[A-Z][\.\)]\s*', '', h)

    # truncate at colon
    if ':' in h:
        h = h.split(':')[0].strip()

    # truncate at comma (catches "includes such factors as interest rates,")
    if ',' in h and len(h.split(',')[0]) > 8:
        h = h.split(',')[0].strip()

    # truncate at em-dash
    h = h.split('—')[0].strip()
    h = h.split('–')[0].strip()

    # kill trailing prepositions: "Economic Environment in" → "Economic Environment"
    h = re.sub(r'\s+(in|of|the|a|an|and|or|for|to|by|as|on)$', '', h, flags=re.IGNORECASE)

    return h.strip()



def extract_headings_from_pdf(pdf_path):
    SKIP_PATTERNS = re.compile(
        r'^\d+$'
        r'|^(fig|figure|table|box)\s*[\d.]'
        r'|^(reprint|isbn|printed|copyright)'
        r'|^[ivxlcdm]+$'
        r'|^activity\s*\d',
        re.IGNORECASE
    )

    try:
        doc = fitz.open(pdf_path)
        all_sizes = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        all_sizes.append(round(span["size"], 1))

        if not all_sizes:
            doc.close()
            return []

        all_sizes.sort()
        median_size = all_sizes[len(all_sizes) // 2]
        heading_threshold = median_size * 1.15

        raw_headings = []
        seen = set()

        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text  = ""
                    is_heading = False
                    for span in line.get("spans", []):
                        size  = span["size"]
                        flags = span["flags"]
                        bold  = bool(flags & 2**4)
                        text  = span["text"].strip()
                        if not text:
                            continue
                        line_text += text + " "
                        if size >= heading_threshold or bold:
                            is_heading = True

                    line_text = line_text.strip()
                    if not line_text or not is_heading:
                        continue
                    if len(line_text) < 4 or len(line_text) > 200:
                        continue
                    if SKIP_PATTERNS.match(line_text):
                        continue

                    key = re.sub(r'\s+', ' ', line_text).lower()
                    if key not in seen:
                        seen.add(key)
                        raw_headings.append(line_text.strip())

        doc.close()
        return raw_headings

    except Exception:
        return []


# ==========================================================
# CLEAN TEXT
# ==========================================================
def clean_text_para(text):
    text = text.replace("\u00ad", "")
    text = NOISE_RE.sub(" ", text)
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n([a-z])', r' \1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    paras = text.split('\n\n')
    paras = [re.sub(r'\s+', ' ', p).strip() for p in paras]
    return '\n\n'.join(p for p in paras if p)

def clean_chunk(text):
    text = text.replace("\u00ad", "")
    text = NOISE_RE.sub(" ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ==========================================================
# LOAD ALL DOCUMENTS
# ==========================================================
@st.cache_resource
def load_documents(pdf_file_list):
    docs  = []
    total = len(pdf_file_list)
    prog  = st.progress(0, text="Reading PDFs...")
    for i, path in enumerate(pdf_file_list):
        raw = read_pdf(path)
        if not raw:
            prog.progress((i+1)/total)
            continue
        cleaned = clean_text_para(raw)
        if len(cleaned.split()) > 200:
            docs.append({
                "doc_id":   os.path.basename(path),
                "text":     cleaned,
            })
        prog.progress((i+1)/total, text=f"Reading {i+1}/{total} — {os.path.basename(path)}")
    prog.empty()
    return docs

# ==========================================================
# STABLE HASH
# ==========================================================
def get_docs_hash(docs):
    fp = [{"doc_id": d["doc_id"], "len": len(d["text"])}
          for d in sorted(docs, key=lambda x: x["doc_id"])]
    return hashlib.md5(json.dumps(fp, sort_keys=True).encode()).hexdigest()

# ==========================================================
# HYBRID CHUNKING
# ==========================================================
def make_overlapping_para_chunks(text, chunk_size, overlap):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraphs = [
        p for p in paragraphs
        if len(p) >= MIN_CHUNK_LEN and not re.match(r'^[\d\s\W]{0,30}$', p)
    ]
    if not paragraphs:
        return []
    chunks = []
    i = 0
    while i < len(paragraphs):
        buf = ""
        j   = i
        while j < len(paragraphs):
            candidate = (buf + "\n\n" + paragraphs[j]).strip() if buf else paragraphs[j]
            if len(candidate) > chunk_size and buf:
                break
            buf = candidate
            j  += 1
        if buf:
            chunks.append(buf)
        target = max(len(buf) - overlap, 0)
        consumed = 0
        i_next = i
        for k in range(i, j):
            consumed += len(paragraphs[k])
            if consumed >= target:
                i_next = k + 1
                break
        else:
            i_next = j
        if i_next <= i:
            i_next = i + 1
        i = i_next
    return chunks

def split_documents(docs):
    chunks = []
    for doc in docs:
        pieces = make_overlapping_para_chunks(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, piece in enumerate(pieces):
            cleaned = clean_chunk(piece)
            if len(cleaned) < MIN_CHUNK_LEN:
                continue
            chunks.append({
                "doc_id":       doc["doc_id"],
                "chunk_id":     f"{doc['doc_id']}_c{idx}",
                "text":         cleaned,
                "section_type": detect_section_type(cleaned),
            })
    return chunks

# ==========================================================
# FAISS INDEX
# ==========================================================
def index_is_valid(h):
    if not all(os.path.exists(p) for p in [INDEX_PATH, META_PATH, HASH_PATH]):
        return False
    with open(HASH_PATH) as f:
        return f.read().strip() == h

def save_index(idx, chunks, h):
    faiss.write_index(idx, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)
    with open(HASH_PATH, "w") as f:
        f.write(h)

def load_index_safe(h):
    try:
        if index_is_valid(h):
            idx = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                chunks = pickle.load(f)
            if idx.ntotal > 0 and len(chunks) > 0:
                return idx, chunks
    except Exception as e:
        st.warning(f"Saved index invalid ({e}), rebuilding...")
        for p in [INDEX_PATH, META_PATH, HASH_PATH]:
            if os.path.exists(p):
                os.remove(p)
    return None, None

@st.cache_resource(show_spinner=False)
def build_index(_docs, docs_hash):
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    idx, chunks = load_index_safe(docs_hash)
    if idx is not None:
        st.success(f"⚡ Index loaded from disk! ({idx.ntotal} vectors, {len(chunks)} chunks)")
        return embed_model, idx, chunks
    chunks = split_documents(_docs)
    if not chunks:
        st.error("No chunks created.")
        st.stop()
    type_counts = {}
    for c in chunks:
        type_counts[c["section_type"]] = type_counts.get(c["section_type"], 0) + 1
    st.info(
        f"📊 {len(chunks)} chunks — " +
        " | ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
    )
    texts    = [c["text"] for c in chunks]
    emb_list = []
    total    = len(texts)
    prog     = st.progress(0, text="Building FAISS index...")
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb   = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        emb_list.append(emb)
        prog.progress(
            min((i + BATCH_SIZE) / total, 1.0),
            text=f"Embedding {min(i+BATCH_SIZE, total)}/{total} chunks..."
        )
    prog.empty()
    embeddings = np.vstack(emb_list).astype("float32")
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    save_index(idx, chunks, docs_hash)
    st.success(f"✅ Index built and saved ({len(chunks)} chunks)")
    return embed_model, idx, chunks

# ==========================================================
# LOAD MODELS
# ==========================================================
@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANK_MODEL_NAME, max_length=512)

# ==========================================================
# OLLAMA
# ==========================================================
@st.cache_resource
def check_ollama():
    if not OLLAMA_URL:
        return False, "OLLAMA_URL not set in config"
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                return True, f"✅ `{OLLAMA_MODEL}` ready"
            else:
                return False, f"`{OLLAMA_MODEL}` not found. Available: {models}"
        return False, f"Ollama returned status {r.status_code}"
    except Exception as e:
        return False, f"Cannot reach Ollama: {e}"

def summarize_with_ollama(topic, top_chunks):
    raw   = " ".join(c["text"] for c in top_chunks)
    words = raw.split()
    if len(words) > 600:
        raw = " ".join(words[:600])

    prompt = f"""You are an expert NCERT exam preparation assistant for Indian students.

Topic: {topic}

NCERT Content:
{raw}

Based ONLY on the above content, write a concise exam flashcard in this exact format:

**Definition:** (one clear sentence defining {topic})

**Key Points:**
1. (first important point)
2. (second important point)
3. (third important point)

**Exam Fact:** (one specific fact about {topic} itself — a definition, number, name, or key distinction that frequently appears in board exam MCQs or 1-mark questions. Must be directly about the main topic, not a subtopic example.)

Keep it factual, based strictly on the provided content."""

    try:
        r = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300}
            },
            timeout=120
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return f"⚠️ Ollama error: status {r.status_code}"
    except requests.exceptions.Timeout:
        best = top_chunks[0]["text"][:400] if top_chunks else ""
        return f"⚠️ *Ollama timed out — try again in 30 seconds.*\n\n**Preview:** {best}"
    except Exception as e:
        best = top_chunks[0]["text"][:400] if top_chunks else ""
        return f"⚠️ *Ollama unavailable.*\n\n**Preview:** {best}"

# ==========================================================
# RETRIEVE + RERANK
# ==========================================================
def retrieve_and_rerank(topic, embed_model, faiss_index, all_chunks, reranker):
    queries = [
        f"What is {topic}?",
        f"Definition and meaning of {topic}",
        f"Overview and introduction to {topic}",
        f"Explain {topic} as defined in NCERT textbooks",
        f"Features characteristics importance of {topic}",
    ]
    q_embs = embed_model.encode(queries, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_embs)
    q_avg  = np.mean(q_embs, axis=0, keepdims=True).astype("float32")
    faiss.normalize_L2(q_avg)
    _, I       = faiss_index.search(q_avg, RETRIEVE_K)
    candidates = [all_chunks[i] for i in I[0] if i < len(all_chunks)]
    if not candidates:
        return [], 0.0
    seen, deduped = [], []
    for c in candidates:
        words  = set(c["text"].lower().split())
        is_dup = any(
            len(words & set(s.split())) / max(len(words | set(s.split())), 1) > 0.6
            for s in seen
        )
        if not is_dup:
            seen.append(c["text"].lower())
            deduped.append(c)
    pairs  = [[topic, c["text"]] for c in deduped]
    scores = reranker.predict(pairs)
    SECTION_BOOST = {
        "noise":      -2.0,
        "definition": +0.8,
        "summary":    +0.3,
        "theory":     +0.1,
        "example":     0.0,
        "activity":   -0.3,
        "exercise":   -0.8,
    }
    boosted = []
    for score, c in zip(scores, deduped):
        adjusted = score + SECTION_BOOST.get(c["section_type"], 0.0)
        boosted.append((adjusted, c))
    boosted.sort(key=lambda x: x[0], reverse=True)
    top_score  = boosted[0][0] if boosted else 0.0
    top_chunks = [c for _, c in boosted[:RERANK_TOP_K]]
    def_chunks   = [c for c in top_chunks if c["section_type"] == "definition"]
    other_chunks = [c for c in top_chunks if c["section_type"] not in ("definition", "noise")]
    return def_chunks + other_chunks, top_score

# ==========================================================
# TOPIC-AWARE FORMATTING
# ==========================================================
def detect_topic_type(topic):
    t = topic.lower()
    if any(w in t for w in ["types", "kinds", "forms", "classification"]):
        return "types"
    elif any(w in t for w in ["difference", "vs", "compare", "distinguish"]):
        return "comparison"
    elif any(w in t for w in ["steps", "process", "procedure", "how"]):
        return "process"
    elif any(w in t for w in ["importance", "significance", "need", "why"]):
        return "importance"
    return "definition"

def format_flashcard(topic, summary, top_score):
    topic_type = detect_topic_type(topic)
    if top_score > 0.70:
        confidence = "🟢 High confidence"
    elif top_score > 0.50:
        confidence = "🟡 Medium confidence"
    else:
        confidence = "🔴 Low confidence — topic may not be well-covered in loaded PDFs"
    headers = {
        "types":      f"### 📘 {topic} — Types & Classification",
        "comparison": f"### 📘 {topic} — Comparison",
        "process":    f"### 📘 {topic} — Process & Steps",
        "importance": f"### 📘 {topic} — Importance & Significance",
        "definition": f"### 📘 {topic} — Concept Summary",
    }
    return f"""
{headers[topic_type]}

{summary}

---
*{confidence} — Reranker score: {top_score:.2f}*
"""

# ==========================================================
# MAIN FLASHCARD FUNCTION
# ==========================================================
def generate_flashcard(topic, embed_model, faiss_index, all_chunks, reranker):
    if not all_chunks:
        return "⚠️ No readable content found."
    top_chunks, top_score = retrieve_and_rerank(topic, embed_model, faiss_index, all_chunks, reranker)
    if not top_chunks:
        return "⚠️ No meaningful content found for this topic."
    with st.expander("🔍 Source Chunks Used (click to inspect)"):
        for i, c in enumerate(c for c in top_chunks if c["section_type"] != "noise"):
            st.markdown(f"**Chunk {i+1}** — `{c['doc_id']}` | type: `{c['section_type']}`")
            st.caption(c["text"][:500] + "..." if len(c["text"]) > 500 else c["text"])
    summary = summarize_with_ollama(topic, top_chunks)
    return format_flashcard(topic, summary, top_score)

# ==========================================================
# ACTIVE LEARNING
# ==========================================================
def extract_keywords_smart(text, top_k=50):
    stopwords = {
        "that", "this", "with", "from", "which", "their", "these", "there",
        "where", "when", "whose", "while", "would", "could", "should", "about",
        "after", "before", "under", "between", "through", "during", "without",
        "within", "although", "because", "however", "therefore", "including",
        "consider", "example", "since", "every", "other", "those", "being"
    }
    freq = {}
    for w in re.findall(r"\b[A-Z][a-z]{3,}\b", text):
        if w.lower() not in stopwords:
            freq[w] = freq.get(w, 0) + 2
    for w in re.findall(r"\b[a-z]{6,}\b", text):
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_k]

def generate_active_learning_card(topic, embed_model, faiss_index, all_chunks, reranker, num_blanks):
    top_chunks, _ = retrieve_and_rerank(topic, embed_model, faiss_index, all_chunks, reranker)
    if not top_chunks:
        return "⚠️ No readable content found.", []
    preferred = [c for c in top_chunks if c["section_type"] in ("definition", "theory")]
    selected  = preferred if preferred else top_chunks
    base_text = " ".join(c["text"] for c in selected[:2])
    base_text = re.sub(r"\s+", " ", base_text).strip()
    keywords = extract_keywords_smart(base_text)
    if num_blanks > len(keywords):
        return (
            f"⚠️ You requested **{num_blanks} blanks**, "
            f"but only **{len(keywords)}** important terms are available.", []
        )
    masked_text    = base_text
    blanks_created = 0
    answer_key     = []
    for kw in keywords:
        if blanks_created >= num_blanks:
            break
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, masked_text):
            masked_text = re.sub(
                pattern,
                f"**[{blanks_created+1}]\\_\\_\\_\\_\\_**",
                masked_text, count=1
            )
            answer_key.append((blanks_created + 1, kw))
            blanks_created += 1
    return f"""
### 🧠 Active Learning: {topic}

**Fill in the missing important terms ({blanks_created} blanks)**

{masked_text}

---
💡 Answers are directly from NCERT content.
""", answer_key

# ==========================================================
# FULL PDF FLASHCARD GENERATION
# ==========================================================
def generate_all_flashcards_for_pdf(pdf_path, pdf_filename, embed_model, faiss_index, all_chunks, reranker):

    SKIP_HEADINGS = {
        "introduction", "summary", "conclusion", "contents", "index",
        "exercises", "review questions", "key terms", "references",
        "bibliography", "appendix", "preface", "foreword", "note",
        "activity", "project", "weblinks", "hints", "answers",
        "very short answer type", "short answer type", "long answer type",
        "multiple choice questions", "fill in the blanks",
    }

    SKIP_FRAGMENTS =  [
    "after studying", "after reading", "you will be able",
    "you would be able", "it enables", "it helps", "it assists",
    "learning objective", "government of india", "enterprise",
    "reprint", "ltd", "corporation",
    "includes such", "includes social", "includes forces",  # ← kills truncated summary bullets
    "includes political", "includes economic",
    "activity i", "activity ii", "activity iii",            # ← kills activity labels
]

    raw_headings = extract_headings_from_pdf(pdf_path)

    cleaned = []
    seen    = set()

    for h in raw_headings:
        c = clean_heading(h)

        # quality filters
        if len(c.split()) < 2:           # at least 2 words
            continue
        if len(c) < 8:                   # at least 8 chars
            continue
        if len(c) > 70:                  # not too long (truncated sentence)
            continue
        if re.match(r'^[a-z]', c):       # must start with capital
            continue
        if c.lower() in SKIP_HEADINGS:
            continue
        if any(f in c.lower() for f in SKIP_FRAGMENTS):
            continue
        if re.search(r'\b(Inc|Ltd|Corp|Pvt)\b', c):  # company names
            continue

        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            cleaned.append(c)

    return cleaned
# ==========================================================
# STARTUP
# ==========================================================
data_path, pdf_files = download_and_extract(FILE_ID)

st.info("📖 Reading PDFs...")
documents = load_documents(tuple(pdf_files))

if not documents:
    st.error("No text could be extracted from the PDFs.")
    st.stop()

st.success(f"✅ Loaded {len(documents)} PDF files")

docs_hash                        = get_docs_hash(documents)
embed_model, faiss_index, chunks = build_index(tuple(documents), docs_hash)
reranker                         = load_reranker()

ollama_ok, ollama_msg = check_ollama()
if ollama_ok:
    st.success(f"✅ Ollama: {ollama_msg}")
else:
    st.error(f"❌ Ollama: {ollama_msg}")

st.success("✅ All models ready!")

# build a mapping: filename → full path (for Tab 3)
pdf_name_to_path = {os.path.basename(p): p for p in pdf_files}

# ==========================================================
# UI
# ==========================================================
tab1, tab2, tab3 = st.tabs(["📘 Flashcard", "🧠 Active Learning", "📚 Full PDF Flashcards"])

# ----------------------------------------------------------
# TAB 1: Single Flashcard
# ----------------------------------------------------------
with tab1:
    topic = st.text_input(
        "Enter Topic (e.g. Fundamental Rights, Business Environment)",
        key="flashcard_topic"
    )
    if st.button("Generate Flashcard"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Retrieving, reranking and summarizing..."):
                st.markdown(generate_flashcard(
                    topic.strip(), embed_model, faiss_index, chunks, reranker
                ))

# ----------------------------------------------------------
# TAB 2: Active Learning
# ----------------------------------------------------------
with tab2:
    topic_al   = st.text_input("Enter Topic for Active Learning", key="active_topic")
    num_blanks = st.number_input(
        "Number of fill-in-the-blanks",
        min_value=1, max_value=25, value=5, step=1
    )
    if st.button("Start Active Learning"):
        if not topic_al.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating active learning card..."):
                result, answer_key = generate_active_learning_card(
                    topic_al.strip(), embed_model, faiss_index, chunks, reranker, int(num_blanks)
                )
                st.markdown(result)
                if answer_key:
                    with st.expander("👁️ Reveal Answer Key"):
                        for num, ans in answer_key:
                            st.write(f"**Blank {num}:** {ans}")

# ----------------------------------------------------------
# TAB 3: Full PDF Flashcards
# ----------------------------------------------------------
with tab3:
    st.subheader("📚 Generate Flashcards for All Topics in a PDF")
    st.caption("Automatically extracts all section headings and generates a flashcard for each.")

    sorted_pdf_names = sorted(pdf_name_to_path.keys())
    selected_pdf = st.selectbox(
        "Select a PDF",
        options=sorted_pdf_names,
        key="pdf_selector"
    )

    if selected_pdf:
        pdf_full_path = pdf_name_to_path[selected_pdf]

        # preview headings before generating
        if st.button("🔍 Preview Topics Found in PDF"):
            with st.spinner("Scanning PDF for headings..."):
                preview_headings = generate_all_flashcards_for_pdf(
                    pdf_full_path, selected_pdf,
                    embed_model, faiss_index, chunks, reranker
                )
            if not preview_headings:
                st.warning("No headings detected in this PDF.")
            else:
                st.success(f"Found **{len(preview_headings)}** topics:")
                for i, h in enumerate(preview_headings, 1):
                    st.write(f"{i}. {h}")

        st.divider()

        # generate all flashcards
        if st.button("⚡ Generate All Flashcards", type="primary"):
            with st.spinner("Scanning headings..."):
                headings = generate_all_flashcards_for_pdf(
                    pdf_full_path, selected_pdf,
                    embed_model, faiss_index, chunks, reranker
                )

            if not headings:
                st.warning("No topics found in this PDF.")
            else:
                st.info(f"Generating flashcards for **{len(headings)}** topics in `{selected_pdf}`... (~{len(headings)*20}s estimated)")

                all_flashcard_text = f"# Flashcards: {selected_pdf}\n\n"
                progress = st.progress(0)
                status   = st.empty()

                for i, heading in enumerate(headings):
                    status.text(f"⏳ {i+1}/{len(headings)}: {heading}")

                    top_chunks_h, top_score_h = retrieve_and_rerank(
                        heading, embed_model, faiss_index, chunks, reranker
                    )

                    if top_chunks_h:
                        summary = summarize_with_ollama(heading, top_chunks_h)
                        card    = format_flashcard(heading, summary, top_score_h)
                    else:
                        card = f"### 📘 {heading}\n\n⚠️ No content found for this topic.\n\n---"

                    # display on screen
                    st.markdown(card)
                    st.divider()

                    # accumulate for download
                    all_flashcard_text += card + "\n\n---\n\n"

                    progress.progress((i + 1) / len(headings))

                status.success(f"✅ Done! Generated {len(headings)} flashcards.")
                progress.empty()

                # download button
                st.download_button(
                    label="⬇️ Download All Flashcards as Markdown",
                    data=all_flashcard_text,
                    file_name=f"flashcards_{selected_pdf.replace('.pdf','')}.md",
                    mime="text/markdown"
                )
