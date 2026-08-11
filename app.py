
import os
import tempfile
from typing import Optional

import requests

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda

from langserve import add_routes


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Job-Ready Career Assistant",
    description=(
        "A personalized career assistant that analyzes a resume, "
        "target role, and GitHub profile."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)


# ============================================================
# PROFILE STORE
# ============================================================
#
# Demo storage.
#
# IMPORTANT:
# This data disappears whenever the Render service restarts.
# For a production application, replace this with PostgreSQL,
# MongoDB, Redis, Supabase, etc.
#
# ============================================================

profile_store = {
    "resume_text": None,
    "skills": None,
    "target_role": None,
    "github_username": None,
    "github_summary": None,
    "github_repositories": None,
}


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RESUME_TEXT = 12000
MAX_PROMPT_RESUME_TEXT = 7000
MAX_GITHUB_REPOS = 10


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="Career-related question"
    )


# ============================================================
# HELPER: EXTRACT PDF TEXT
# ============================================================

def extract_resume_text(pdf_bytes: bytes) -> str:
    """
    Extract text from an uploaded PDF.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    temp_path: Optional[str] = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(pdf_bytes)
            temp_path = tmp.name

        reader = PdfReader(temp_path)

        if not reader.pages:
            raise ValueError("The PDF contains no pages.")

        text_parts = []

        for page in reader.pages:

            page_text = page.extract_text() or ""

            if page_text.strip():
                text_parts.append(page_text)

        text = "\n".join(text_parts).strip()

        if not text:
            raise ValueError(
                "Could not extract text from the PDF. "
                "The resume may be scanned/image-based."
            )

        return text[:MAX_RESUME_TEXT]

    finally:

        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ============================================================
# HELPER: EXTRACT SKILLS USING GEMINI
# ============================================================

def extract_skills(resume_text: str) -> str:
    """
    Extract technical skills from the resume.
    """

    prompt = f"""
You are a resume parsing system.

Extract ONLY technical skills from the resume.

Include:
- Programming languages
- Frameworks
- Libraries
- Databases
- Cloud platforms
- AI/ML technologies
- Developer tools
- APIs
- Software technologies
- Relevant platforms

Do NOT include:
- Soft skills
- Communication skills
- Leadership
- Hobbies
- Personal traits
- Job titles
- Company names

Return ONLY a clean comma-separated list.

Resume:
{resume_text[:MAX_RESUME_TEXT]}
"""

    response = llm.invoke(prompt)

    skills = response.content.strip()

    if not skills:
        return "No technical skills detected."

    return skills


# ============================================================
# HELPER: FETCH GITHUB PROFILE
# ============================================================

def fetch_github_profile(username: str):
    """
    Fetch GitHub profile and repository information.
    """

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Job-Ready-Career-Assistant",
    }

    profile_url = (
        f"https://api.github.com/users/{username}"
    )

    try:

        response = requests.get(
            profile_url,
            headers=headers,
            timeout=10,
        )

        if response.status_code == 404:
            return (
                "GitHub username not found.",
                []
            )

        if response.status_code != 200:
            return (
                f"GitHub API returned status "
                f"{response.status_code}.",
                []
            )

        user = response.json()

        github_summary = (
            f"GitHub username: {username}\n"
            f"Name: {user.get('name') or 'Not provided'}\n"
            f"Public repositories: "
            f"{user.get('public_repos', 0)}\n"
            f"Followers: {user.get('followers', 0)}\n"
            f"Following: {user.get('following', 0)}\n"
            f"Bio: {user.get('bio') or 'Not provided'}\n"
            f"Profile URL: "
            f"{user.get('html_url') or 'Not available'}"
        )

        # ----------------------------------------------------
        # Fetch repositories
        # ----------------------------------------------------

        repos_url = (
            f"https://api.github.com/users/"
            f"{username}/repos"
        )

        repos_response = requests.get(
            repos_url,
            headers=headers,
            params={
                "sort": "updated",
                "direction": "desc",
                "per_page": MAX_GITHUB_REPOS,
            },
            timeout=10,
        )

        repositories = []

        if repos_response.status_code == 200:

            repos = repos_response.json()

            for repo in repos:

                repositories.append({
                    "name": repo.get("name"),
                    "description": (
                        repo.get("description")
                        or "No description"
                    ),
                    "language": (
                        repo.get("language")
                        or "Not specified"
                    ),
                    "stars": repo.get(
                        "stargazers_count",
                        0
                    ),
                    "url": repo.get("html_url"),
                })

        return github_summary, repositories

    except requests.RequestException:

        return (
            "GitHub profile could not be fetched "
            "because of a network error.",
            []
        )

    except Exception:

        return (
            "GitHub profile could not be processed.",
            []
        )


# ============================================================
# HELPER: BUILD GITHUB REPOSITORY TEXT
# ============================================================

def format_github_repositories(repositories) -> str:

    if not repositories:
        return "No public repositories available."

    lines = []

    for repo in repositories:

        lines.append(
            f"- {repo['name']} | "
            f"Language: {repo['language']} | "
            f"Stars: {repo['stars']} | "
            f"Description: {repo['description']}"
        )

    return "\n".join(lines)


# ============================================================
# UPLOAD PROFILE
# ============================================================

@app.post("/upload_profile")
async def upload_profile(
    resume: UploadFile = File(...),
    target_role: str = Form(...),
    github_username: str = Form(...),
):
    """
    Upload resume and create the candidate profile.
    """

    # --------------------------------------------------------
    # Validate target role
    # --------------------------------------------------------

    target_role = target_role.strip()

    if not target_role:
        raise HTTPException(
            status_code=400,
            detail="Target role cannot be empty."
        )

    # --------------------------------------------------------
    # Validate GitHub username
    # --------------------------------------------------------

    github_username = github_username.strip()

    if not github_username:
        raise HTTPException(
            status_code=400,
            detail="GitHub username cannot be empty."
        )

    # --------------------------------------------------------
    # Validate PDF
    # --------------------------------------------------------

    filename = resume.filename or ""

    if not filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF resume."
        )

    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    try:

        pdf_bytes = await resume.read()

        resume_text = extract_resume_text(
            pdf_bytes
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Resume processing failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Extract skills
    # --------------------------------------------------------

    try:

        skills = extract_skills(
            resume_text
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Skill extraction failed: {str(e)}"
        )

    # --------------------------------------------------------
    # GitHub
    # --------------------------------------------------------

    github_summary, repositories = (
        fetch_github_profile(
            github_username
        )
    )

    # --------------------------------------------------------
    # Store profile
    # --------------------------------------------------------

    profile_store["resume_text"] = resume_text
    profile_store["skills"] = skills
    profile_store["target_role"] = target_role
    profile_store["github_username"] = github_username
    profile_store["github_summary"] = github_summary
    profile_store["github_repositories"] = repositories

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "message": "Profile uploaded successfully.",
        "target_role": target_role,
        "skills": skills,
        "github_username": github_username,
        "github_summary": github_summary,
        "github_repositories": repositories,
    }


# ============================================================
# GET CURRENT PROFILE
# ============================================================

@app.get("/profile")
def get_profile():
    """
    Return the currently loaded candidate profile.
    """

    if profile_store["skills"] is None:

        return {
            "profile_loaded": False,
            "message": "No profile has been uploaded yet."
        }

    return {
        "profile_loaded": True,
        "target_role": profile_store["target_role"],
        "skills": profile_store["skills"],
        "github_username": profile_store["github_username"],
        "github_summary": profile_store["github_summary"],
        "github_repositories": (
            profile_store["github_repositories"]
        ),
    }


# ============================================================
# CAREER ASSISTANT CORE
# ============================================================

async def run_career_assistant(
    question: str
) -> str:
    """
    Core personalized career assistant.
    """

    # --------------------------------------------------------
    # Make sure profile exists
    # --------------------------------------------------------

    if profile_store["skills"] is None:

        return (
            "Please upload your resume, target role, "
            "and GitHub username first using "
            "/upload_profile."
        )

    # --------------------------------------------------------
    # GitHub repositories
    # --------------------------------------------------------

    github_repositories = (
        format_github_repositories(
            profile_store["github_repositories"]
            or []
        )
    )

    # --------------------------------------------------------
    # Personalized prompt
    # --------------------------------------------------------

    prompt = f"""
You are a personalized Job-Ready Career Assistant.

Your job is to help the candidate become job-ready
for their target role.

==============================
CANDIDATE PROFILE
==============================

Target role:
{profile_store["target_role"]}

Technical skills:
{profile_store["skills"]}

GitHub profile:
{profile_store["github_summary"]}

Recent GitHub repositories:
{github_repositories}

Resume:
{profile_store["resume_text"][:MAX_PROMPT_RESUME_TEXT]}

==============================
USER QUESTION
==============================

{question}

==============================
INSTRUCTIONS
==============================

1. Answer specifically for THIS candidate.

2. Use the candidate's current skills, resume,
   target role, and GitHub projects.

3. Do not invent skills, projects, education,
   experience, or achievements.

4. If the candidate is missing important skills,
   clearly identify those skill gaps.

5. If the question asks for a roadmap,
   prioritize the most important topics first.

6. If the question asks whether the candidate
   is job-ready, give an honest assessment.

7. If the candidate has relevant GitHub projects,
   explain how those projects can strengthen
   their profile.

8. Prefer practical recommendations over generic advice.

9. When useful, organize the answer using:
   - Current status
   - Skill gaps
   - What to learn
   - Projects to build
   - Next steps

10. Keep answers concise but useful.

11. Never claim that the candidate knows something
    unless it appears in the provided profile.

Answer the user's question now.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# CHAT API
# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Standard JSON chat endpoint.
    """

    answer = await run_career_assistant(
        req.question
    )

    return {
        "answer": answer
    }


# ============================================================
# LANGSERVE PLAYGROUND
# ============================================================
#
# This accepts plain text from:
#
# /agent/playground/
#
# Example:
#
# "What should I learn next?"
#
# ============================================================

async def playground_chat(
    text: str
) -> str:

    text = text.strip()

    if not text:

        return (
            "Please enter a career-related question."
        )

    return await run_career_assistant(
        text
    )


clean_agent = RunnableLambda(
    playground_chat
)


add_routes(
    app,
    clean_agent,
    path="/agent"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Job-Ready Career Assistant"
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Job-Ready Career Assistant is running.",
        "docs": "/docs",
        "playground": "/agent/playground/",
        "health": "/health",
        "upload_profile": "/upload_profile",
        "chat": "/chat",
        "profile": "/profile",
    }

