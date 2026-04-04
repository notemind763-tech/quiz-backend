import os
import json
import io
import re
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    import pdfplumber
    PDF_ENGINE = "pdfplumber"
except ImportError:
    from pypdf import PdfReader
    PDF_ENGINE = "pypdf"

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quizlab")

app = FastAPI(title="QuizLab API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_MB = 20
MAX_TEXT_CHARS = 60000

SYSTEM_PROMPT = """You are a precise MCQ extractor. Extract multiple-choice questions from text.
Rules:
- Only extract real questions with 4 options (A, B, C, D)
- If the answer key is missing, make your best guess based on context
- Assign difficulty 1-5: 1=very easy, 3=medium, 5=very hard
- Assign sec (section/topic) based on content
- Output ONLY a valid JSON array, no extra text"""

USER_PROMPT_TEMPLATE = """Extract all MCQ questions from the text below.

Output strictly as JSON array:
[
  {{
    "q": "Full question text here?",
    "o": {{"A": "Option text", "B": "Option text", "C": "Option text", "D": "Option text"}},
    "c": "B",
    "e": "Brief explanation of why B is correct",
    "d": 3,
    "sec": "Topic/Section Name",
    "sub": "Specific subtopic"
  }}
]

TEXT:
{text}

Return ONLY the JSON array. No markdown, no explanation."""


def extract_pdf_text(file_bytes: bytes) -> str:
    text_parts = []
    if PDF_ENGINE == "pdfplumber":
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:100]:
                t = page.extract_text(x_tolerance=2, y_tolerance=2)
                if t:
                    text_parts.append(t)
    else:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages[:100]:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    full = "\n".join(text_parts)
    full = re.sub(r'[ \t]{2,}', ' ', full)
    full = re.sub(r'\n{3,}', '\n\n', full)
    return full.strip()


def chunk_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    overlap = 500
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        last_break = chunk.rfind('\n\n')
        if last_break > max_chars * 0.6:
            chunk = chunk[:last_break]
            end = start + last_break
        chunks.append(chunk)
        start = end - overlap
    return chunks


def call_gemini(text: str) -> list:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(text=text),
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192
        )
    )
    raw = response.text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        for key in ("questions", "mcqs", "data", "items", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return parsed if isinstance(parsed, list) else []

def validate_and_clean(questions: list) -> list:
    cleaned = []
    seen = set()
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        q_text = str(q.get("q", "")).strip()
        options = q.get("o", {})
        correct = str(q.get("c", "")).strip().upper()
        if len(q_text) < 8:
            continue
        if not isinstance(options, dict) or len(options) < 2:
            continue
        if correct not in options:
            m = re.search(r'[ABCD]', correct)
            if m and m.group() in options:
                correct = m.group()
            else:
                continue
        key = q_text[:50].lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "id": f"q_{int(time.time()*1000)}_{i}",
            "q": q_text[:500],
            "o": {k.upper(): str(v)[:200] for k, v in options.items()},
            "c": correct,
            "e": str(q.get("e", ""))[:400],
            "oe": q.get("oe", {}),
            "d": max(1, min(5, int(q.get("d", 3)))),
            "sec": str(q.get("sec", "General"))[:60],
            "sub": str(q.get("sub", "Imported"))[:60],
        })
    return cleaned


@app.get("/")
def root():
    return {"status": "ok", "service": "QuizLab API", "version": "3.0.0", "pdf_engine": PDF_ENGINE}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/api/extract-questions")
async def extract_questions(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max is {MAX_FILE_MB}MB.")
    if len(contents) < 100:
        raise HTTPException(400, "File is empty or corrupted.")
    logger.info(f"📄 PDF received: {file.filename} ({size_mb:.2f}MB)")
    start = time.time()
        try:
        full_text = extract_pdf_text(contents)
        logger.info(f"✅ Text extracted: {len(full_text)} chars")
    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}", exc_info=True)
        raise HTTPException(422, f"Could not read PDF: {str(e)}")
    if not full_text or len(full_text) < 50:
        raise HTTPException(422, "PDF has no readable text.")
    logger.info(f"📝 Extracted {len(full_text)} chars")
    chunks = chunk_text(full_text, MAX_TEXT_CHARS)
    logger.info(f"🔀 Split into {len(chunks)} chunk(s)")
    all_questions = []
    warnings = []
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"🤖 Processing chunk {chunk_idx + 1}/{len(chunks)}")
        try:
            raw_qs = call_gemini(chunk)
            all_questions.extend(raw_qs)
        except Exception as e:
            logger.error(f"Chunk {chunk_idx + 1} error: {e}")
            warnings.append(f"Chunk {chunk_idx + 1} error: {str(e)}")
        if chunk_idx < len(chunks) - 1:
            time.sleep(1)
    final_questions = validate_and_clean(all_questions)
    elapsed = round(time.time() - start, 2)
    logger.info(f"✅ Done: {len(final_questions)} questions in {elapsed}s")
    if not final_questions:
        warnings.append("No valid MCQ questions found.")
    return {
        "success": True,
        "count": len(final_questions),
        "data": final_questions,
        "processing_time_seconds": elapsed,
        "pdf_engine": PDF_ENGINE,
        "warnings": warnings,
    }


@app.post("/api/parse-text")
async def parse_text(request: Request):
    try:
        body = await request.json()
        raw_text = body.get("text", "")
        if not raw_text or len(raw_text) < 20:
            raise HTTPException(400, "No text provided.")
        raw_qs = call_gemini(raw_text[:8000])
        final = validate_and_clean(raw_qs)
        return {"success": True, "count": len(final), "data": final}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"parse-text error: {e}")
        raise HTTPException(500, str(e))


@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Server error. Please try again."})
