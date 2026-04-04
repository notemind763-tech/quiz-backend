
import os
import json
import io
import re
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Use pdfplumber for better text extraction (add to requirements.txt)
try:
    import pdfplumber
    PDF_ENGINE = "pdfplumber"
except ImportError:
    from pypdf import PdfReader
    PDF_ENGINE = "pypdf"

from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quizlab")

app = FastAPI(title="QuizLab API", version="2.1.0")

# ── CORS ─────────────────────────────────────────────────────────────────────
# Replace the Netlify URL with your actual domain
ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "https://*.netlify.app",
    # ✅ Add your exact Netlify URL here:
    # "https://your-quizlab.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Change to ALLOWED_ORIGINS after testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────────────────────
MAX_FILE_MB   = 20
MAX_TEXT_CHARS = 30000   # Per Groq context window — we chunk if longer
GROQ_MODEL    = "llama-3.3-70b-versatile"   # Better accuracy than 8b; still free
FALLBACK_MODEL = "llama-3.1-8b-instant"  # Used if 70b quota hit


# ── PDF Text Extraction ──────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text using best available engine."""
    text_parts = []

    if PDF_ENGINE == "pdfplumber":
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:200]:
                t = page.extract_text(x_tolerance=2, y_tolerance=2)
                if t:
                    text_parts.append(t)
    else:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages[:200]:
            t = page.extract_text()
            if t:
                text_parts.append(t)

    full = "\n".join(text_parts)

    # Clean OCR noise
    full = re.sub(r'[ \t]{2,}', ' ', full)
    full = re.sub(r'\n{3,}', '\n\n', full)

    return full.strip()


def chunk_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> list[str]:
    """
    Split long text into overlapping chunks so we don't miss questions
    that span the cutoff boundary.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    overlap = 500   # Overlap to avoid cutting mid-question
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        # Try to cut at a paragraph boundary
        last_break = chunk.rfind('\n\n')
        if last_break > max_chars * 0.6:
            chunk = chunk[:last_break]
            end = start + last_break
        chunks.append(chunk)
        start = end - overlap
    return chunks


# ── Groq Prompt ──────────────────────────────────────────────────────────────
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


def call_groq(client: Groq, text: str, model: str) -> list:
    """Call Groq and parse the JSON response safely."""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        model=model,
        temperature=0.1,      # Low temp = more consistent output
        max_tokens=4096,
        # Note: response_format json_object forces a dict, not array
        # So we parse manually for array support
    )
    raw = response.choices[0].message.content.strip()
    logger.info(f"Groq raw response length: {len(raw)} chars")

    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Try direct parse
    parsed = json.loads(raw)

    # Handle {"questions": [...]} wrapper
    if isinstance(parsed, dict):
        for key in ("questions", "mcqs", "data", "items", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # Take the first list value
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

    return parsed if isinstance(parsed, list) else []


def validate_and_clean(questions: list) -> list:
    """Validate each question has required fields and clean values."""
    cleaned = []
    seen = set()

    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue

        q_text = str(q.get("q", "")).strip()
        options = q.get("o", {})
        correct = str(q.get("c", "")).strip().upper()

        # Must have: question text, at least 2 options, correct answer
        if len(q_text) < 8:
            continue
        if not isinstance(options, dict) or len(options) < 2:
            continue
        if correct not in options:
            # Try to recover if correct is like "A." or "Option A"
            m = re.search(r'[ABCD]', correct)
            if m and m.group() in options:
                correct = m.group()
            else:
                continue  # Skip unrecoverable

        # Deduplicate by first 50 chars of question
        key = q_text[:50].lower()
        if key in seen:
            continue
        seen.add(key)

        cleaned.append({
            "id": f"q_{int(time.time()*1000)}_{i}",
            "q":   q_text[:500],
            "o":   {k.upper(): str(v)[:200] for k, v in options.items()},
            "c":   correct,
            "e":   str(q.get("e", ""))[:400],
            "oe":  q.get("oe", {}),
            "d":   max(1, min(5, int(q.get("d", 3)))),
            "sec": str(q.get("sec", "General"))[:60],
            "sub": str(q.get("sub", "Imported"))[:60],
        })

    return cleaned


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "QuizLab API", "version": "2.1.0", "pdf_engine": PDF_ENGINE}


@app.get("/health")
def health():
    """Ping this every 14 min with UptimeRobot to prevent Vercel cold starts."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/api/extract-questions")
async def extract_questions(file: UploadFile = File(...)):
    # ── 1. Validate file ─────────────────────────────────────
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

    # ── 2. Extract text ──────────────────────────────────────
    try:
        full_text = extract_pdf_text(contents)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(422, f"Could not read PDF. Is it a scanned image? Error: {str(e)}")

    if not full_text or len(full_text) < 50:
        raise HTTPException(422, "PDF has no readable text. If it's a scanned PDF, text extraction won't work.")

    logger.info(f"📝 Extracted {len(full_text)} chars of text")

    # ── 3. Get Groq client ───────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(500, "GROQ_API_KEY environment variable is not set on the server.")

    client = Groq(api_key=api_key)

    # ── 4. Chunk + call Groq ─────────────────────────────────
    chunks = chunk_text(full_text, MAX_TEXT_CHARS)
    logger.info(f"🔀 Split into {len(chunks)} chunk(s)")

    all_questions = []
    warnings = []

    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"🤖 Processing chunk {chunk_idx + 1}/{len(chunks)} with {GROQ_MODEL}")
        try:
            raw_qs = call_groq(client, chunk, GROQ_MODEL)
            all_questions.extend(raw_qs)
        except json.JSONDecodeError as e:
            logger.warning(f"Chunk {chunk_idx + 1}: JSON parse failed, trying fallback model")
            try:
                raw_qs = call_groq(client, chunk, FALLBACK_MODEL)
                all_questions.extend(raw_qs)
            except Exception as e2:
                warnings.append(f"Chunk {chunk_idx + 1} failed: {str(e2)}")
        except Exception as e:
            logger.error(f"Chunk {chunk_idx + 1} error: {e}")
            warnings.append(f"Chunk {chunk_idx + 1} error: {str(e)}")

        # Groq rate limit: 30 req/min on free tier — add small delay between chunks
        if chunk_idx < len(chunks) - 1:
            time.sleep(1)

    # ── 5. Validate + deduplicate ────────────────────────────
    final_questions = validate_and_clean(all_questions)

    elapsed = round(time.time() - start, 2)
    logger.info(f"✅ Done: {len(final_questions)} questions in {elapsed}s")

    if not final_questions:
        warnings.append("No valid MCQ questions found. Ensure the PDF contains questions with A/B/C/D options.")

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
    body = await request.json()
    raw_text = body.get("text", "")
    if not raw_text or len(raw_text) < 20:
        raise HTTPException(400, "No text provided.")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(500, "GROQ_API_KEY not set.")
    
    client = Groq(api_key=api_key)
    raw_qs = call_groq(client, raw_text[:8000], GROQ_MODEL)
    final = validate_and_clean(raw_qs)
    
    return {"success": True, "count": len(final), "data": final}


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Server error. Please try again."})
