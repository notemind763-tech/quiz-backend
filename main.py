import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq

app = FastAPI()

# Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # We will allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "awake"}

@app.post("/api/extract-questions")
async def extract_questions(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    # Get API key from Render environment
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key is missing on the server.")
    
    client = Groq(api_key=api_key)

    try:
        # 1. Read PDF Text locally (fast & free)
        reader = PdfReader(file.file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")

        # 2. Send text to Groq AI for structuring
        prompt = f"""
        Extract multiple-choice questions from the following text. 
        Format the output strictly as a JSON array of objects. 
        Determine a difficulty (1-5) and a topic/section for each.
        Use exactly this JSON schema:
        [{{
            "q": "Question text",
            "o": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
            "c": "A", 
            "e": "Explanation of the answer",
            "d": 3,
            "sec": "Subject/Category",
            "sub": "Specific Topic"
        }}]
        
        Text to parse:
        {text[:15000]}
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", 
            response_format={"type": "json_object"},
        )

        raw_json = response.choices[0].message.content
        parsed_data = json.loads(raw_json)
        
        # Ensure it returns an array
        if isinstance(parsed_data, dict):
            # If the AI returns {"questions": [...]}, extract the list
            key = list(parsed_data.keys())[0]
            parsed_data = parsed_data[key]

        return {"success": True, "data": parsed_data}

    except Exception as e:
        print("Backend Error:", str(e))
        raise HTTPException(status_code=500, detail="AI parsing failed. " + str(e))