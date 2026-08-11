import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="Job-Ready Career Assistant")

# Gemini model
llm = ChatGoogleGenerativeAI(model="models/gemini-3.5-flash-lite")

# Temporary in-memory storage
stored_resume = {
    "skills": None,
    "text": None
}

# -------------------- Upload Resume --------------------

@app.post("/upload_resume")
async def upload_resume(resume: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    os.unlink(tmp_path)

    prompt = f"""
Extract ONLY the technical skills, tools, programming languages,
frameworks, databases, and platforms mentioned in this resume.
Return a comma-separated list.

Resume:
{text[:6000]}
"""

    response = llm.invoke(prompt)
    skills = str(response.content).strip()

    stored_resume["skills"] = skills
    stored_resume["text"] = text

    return {
        "message": "Resume uploaded successfully",
        "skills": skills
    }

# -------------------- Chat Endpoint --------------------

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest):

    if stored_resume["skills"] is None:
        return {
            "answer": "Please upload your resume first using /upload_resume."
        }

    prompt = f"""
You are a job-ready career assistant.

Candidate skills:
{stored_resume['skills']}

Resume summary:
{stored_resume['text'][:4000]}

User question:
{req.question}

Answer specifically for this candidate.
Give practical and personalized guidance.
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}

# -------------------- Health --------------------

@app.get("/")
def root():
    return {"message": "Job-Ready Career Assistant is running"}
