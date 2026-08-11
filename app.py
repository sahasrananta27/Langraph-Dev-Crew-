import os
import tempfile
import requests

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="Job-Ready Career Assistant")

llm = ChatGoogleGenerativeAI(model="models/gemini-3.5-flash-lite")

# In-memory profile store (demo purpose)
profile_store = {
    "resume_text": None,
    "skills": None,
    "target_role": None,
    "github_username": None,
    "github_summary": None,
}

# -------------------- Upload Profile --------------------

@app.post("/upload_profile")
async def upload_profile(
    resume: UploadFile = File(...),
    target_role: str = Form(...),
    github_username: str = Form(...),
):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    # Extract text
    reader = PdfReader(tmp_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    os.unlink(tmp_path)

    # Extract skills using Gemini
    prompt = f"""
Extract ONLY technical skills, tools, programming languages,
frameworks, databases, and platforms from this resume.
Return a comma-separated list.

Resume:
{text[:6000]}
"""

    skills = llm.invoke(prompt).content.strip()

    # GitHub summary
    github_summary = "GitHub profile could not be fetched."

    try:
        res = requests.get(f"https://api.github.com/users/{github_username}")

        if res.status_code == 200:
            user = res.json()
            github_summary = (
                f"GitHub: {github_username}, "
                f"Public repos: {user.get('public_repos')}, "
                f"Followers: {user.get('followers')}, "
                f"Bio: {user.get('bio') or 'None'}"
            )
    except Exception:
        pass

    # Store profile
    profile_store["resume_text"] = text
    profile_store["skills"] = skills
    profile_store["target_role"] = target_role
    profile_store["github_username"] = github_username
    profile_store["github_summary"] = github_summary

    return {
        "message": "Profile uploaded successfully",
        "target_role": target_role,
        "skills": skills,
        "github_summary": github_summary,
    }

# -------------------- Chat --------------------

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest):

    if profile_store["skills"] is None:
        return {
            "answer": "Please upload your resume, target role, and GitHub username first using /upload_profile."
        }

    prompt = f"""
You are a personalized Job-Ready Career Assistant.

Candidate target role:
{profile_store['target_role']}

Candidate skills:
{profile_store['skills']}

GitHub summary:
{profile_store['github_summary']}

Resume summary:
{profile_store['resume_text'][:5000]}

User question:
{req.question}

Answer specifically for THIS candidate.
Be practical, personalized, and concise.
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}

# -------------------- Root --------------------

@app.get("/")
def root():
    return {"message": "Job-Ready Career Assistant is running"}
