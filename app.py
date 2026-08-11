import os
import tempfile
import requests

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# LLM
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.3
)


# ============================================================
# PROFILE STORAGE
# ============================================================

profile = {
    "resume": "",
    "skills": "",
    "role": "",
    "github": "",
    "github_data": ""
}


# ============================================================
# HELPERS
# ============================================================

def text_from_response(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        result = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    result.append(item.get("text", ""))
            else:
                result.append(str(item))

        return "\n".join(result).strip()

    return str(content).strip()


def extract_pdf(file_bytes):
    if not file_bytes:
        raise ValueError("Empty PDF file.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    try:
        reader = PdfReader(path)

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        if not text.strip():
            raise ValueError(
                "Could not extract text from PDF. "
                "The resume may be scanned."
            )

        return text[:12000]

    finally:
        if os.path.exists(path):
            os.unlink(path)


def get_github(username):
    try:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "PlacementGuider"
        }

        token = os.getenv("GITHUB_TOKEN")

        if token:
            headers["Authorization"] = f"Bearer {token}"

        user = requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers,
            timeout=10
        )

        if user.status_code != 200:
            return "GitHub profile not found."

        data = user.json()

        repos_response = requests.get(
            f"https://api.github.com/users/{username}/repos",
            headers=headers,
            params={
                "sort": "updated",
                "per_page": 10
            },
            timeout=10
        )

        repos = (
            repos_response.json()
            if repos_response.status_code == 200
            else []
        )

        languages = {}

        for repo in repos:
            language = repo.get("language")

            if language:
                languages[language] = (
                    languages.get(language, 0) + 1
                )

        return f"""
GitHub username: {username}
Public repositories: {data.get("public_repos", 0)}
Followers: {data.get("followers", 0)}
Bio: {data.get("bio") or "Not provided"}
Top languages: {languages}

Recent repositories:
{
    chr(10).join(
        f"- {r.get('name')}: "
        f"{r.get('description') or 'No description'} "
        f"({r.get('language') or 'Unknown'})"
        for r in repos
    )
}
""".strip()

    except Exception as e:
        return f"GitHub analysis unavailable: {str(e)}"


# ============================================================
# PROFILE ANALYSIS
# ============================================================

def analyze_profile():

    prompt = f"""
You are PlacementGuider, a professional career advisor.

Analyze this candidate.

TARGET ROLE:
{profile["role"]}

RESUME:
{profile["resume"][:9000]}

EXTRACTED TECHNICAL SKILLS:
{profile["skills"]}

GITHUB:
{profile["github_data"]}

Give a practical analysis.

Use exactly these sections:

## 1. Target Role Readiness
Say whether the candidate is:
- Ready
- Almost Ready
- Needs More Preparation

Explain briefly why.

## 2. Current Skills
List the strongest relevant skills from the resume.

## 3. Skill Gaps
List the most important skills the candidate should learn
for the target role.

Prioritize them.

## 4. Other Suitable Roles
Suggest 3-5 other roles the candidate could realistically
consider based on their current skills.

## 5. Learning Roadmap
Give a practical roadmap:

Phase 1 - Fundamentals
Phase 2 - Role-specific skills
Phase 3 - Projects
Phase 4 - Interview preparation

## 6. Project Recommendations
Suggest 2-3 projects that would improve the candidate's
profile for the target role.

## 7. GitHub Improvements
Suggest improvements to the GitHub profile and repositories.

## 8. Resume Improvements
Suggest specific improvements to:
- Skills section
- Projects
- Resume structure
- Missing keywords
- Achievement descriptions

Do NOT invent experience or skills.

Be honest and practical.
"""


    return text_from_response(
        llm.invoke(prompt)
    )


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(question):

    if not profile["resume"]:
        return (
            "Please upload your resume, target role, "
            "and GitHub username first."
        )

    question = question.strip()

    if not question:
        return "Please ask a career-related question."

    # --------------------------------------------------------
    # Career-topic validation
    # --------------------------------------------------------

    check_prompt = f"""
You are a strict career-question classifier.

Determine whether the user's question is related to
career, jobs, placements, resumes, skills, programming
skills, learning, projects, GitHub, interviews, job roles,
career roadmap, salary preparation, or professional growth.

User question:
{question}

Return ONLY one word:

CAREER

or

NOT_CAREER
"""

    try:
        check_response = llm.invoke(check_prompt)

        classification = text_from_response(
            check_response
        ).strip().upper()

    except Exception:
        return (
            "Sorry, I can only help with career-related "
            "questions based on your uploaded profile."
        )

    # --------------------------------------------------------
    # Reject unrelated questions
    # --------------------------------------------------------

    if "CAREER" not in classification:
        return (
            "Sorry, I can only answer career-related "
            "questions based on your uploaded resume, "
            "target role, skills, and GitHub profile."
        )

    # --------------------------------------------------------
    # Career question
    # --------------------------------------------------------

    prompt = f"""
You are PlacementPrep, a personalized career assistant.

You MUST ONLY discuss the candidate's career,
education-to-career preparation, technical skills,
jobs, placements, resumes, projects, GitHub,
interviews, learning roadmap, and professional growth.

Candidate profile:

TARGET ROLE:
{profile["role"]}

RESUME:
{profile["resume"][:8000]}

CURRENT TECHNICAL SKILLS:
{profile["skills"]}

GITHUB:
{profile["github_data"]}

USER QUESTION:
{question}

Rules:

1. Answer ONLY the career-related question.

2. Base your answer on the candidate's uploaded profile.

3. Do not invent skills, projects, education,
   experience, certifications, or achievements.

4. If the candidate lacks a required skill,
   clearly identify it.

5. If the candidate asks whether they can achieve
   a role, honestly compare their current skills
   with the role requirements.

6. If they ask what to learn, prioritize the
   most important missing skills.

7. If they ask for a roadmap, give an ordered
   learning roadmap.

8. If they ask about their resume, give specific
   resume improvements.

9. If they ask about GitHub, analyze their
   GitHub information.

10. If they ask about other suitable jobs,
    recommend roles based on their profile.

11. Never answer unrelated questions.

12. Never engage in casual conversation,
    romance, jokes, entertainment, general trivia,
    weather, politics, or unrelated topics.

Give a practical and concise answer.
"""

    try:

        response = llm.invoke(prompt)

        return text_from_response(response)

    except Exception as e:

        return (
            "Sorry, I couldn't analyze that career "
            "question right now. Please try again."
        )
# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PlacementPrep Career Assistant",
    version="2.0",
    description="Personalized resume and career analysis"
)


# ============================================================
# HOME UI
# ============================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

<title>PlacementPrep</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

body {
    font-family: Arial;
    background: #f1f5f9;
    margin: 0;
}

.container {
    max-width: 800px;
    margin: 40px auto;
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 5px 25px #0002;
}

h1 {
    text-align: center;
    color: #2563eb;
}

.subtitle {
    text-align: center;
    color: #64748b;
}

label {
    display: block;
    margin-top: 18px;
    font-weight: bold;
}

input {
    width: 100%;
    padding: 12px;
    margin-top: 7px;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    box-sizing: border-box;
}

button {
    width: 100%;
    padding: 13px;
    margin-top: 25px;
    border: none;
    border-radius: 7px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

#loading {
    display: none;
    text-align: center;
    margin-top: 20px;
}

#result {
    display: none;
    margin-top: 30px;
}

#skills {
    background: #eff6ff;
    padding: 15px;
    border-radius: 8px;
}

#report {
    white-space: pre-wrap;
    line-height: 1.6;
}

.error {
    color: #dc2626;
    margin-top: 15px;
}

</style>

</head>

<body>

<div class="container">

<h1>🎯 PlacementPrep</h1>

<p class="subtitle">
Personalized Resume & Career Assistant
</p>

<label>📄 Resume PDF</label>

<input
    type="file"
    id="resume"
    accept=".pdf"
>

<label>💼 Target Role</label>

<input
    type="text"
    id="role"
    placeholder="Example: Python Developer"
>

<label>🐙 GitHub Username</label>

<input
    type="text"
    id="github"
    placeholder="Example: sahasra123"
>

<button onclick="analyze()">
Analyze My Profile
</button>

<div id="loading">
⏳ Analyzing resume, skills and GitHub...
</div>

<div id="error" class="error"></div>

<div id="result">

<h2>🛠 Current Skills</h2>

<div id="skills"></div>

<h2>📊 Career Analysis</h2>

<div id="report"></div>

</div>

</div>

<script>

async function analyze() {

    const resume =
        document.getElementById("resume").files[0];

    const role =
        document.getElementById("role").value.trim();

    const github =
        document.getElementById("github").value.trim();

    if (!resume || !role || !github) {

        document.getElementById("error").innerText =
            "Please provide resume, target role and GitHub username.";

        return;
    }

    const form = new FormData();

    form.append("resume", resume);
    form.append("role", role);
    form.append("github_username", github);

    document.getElementById("loading").style.display = "block";
    document.getElementById("result").style.display = "none";
    document.getElementById("error").innerText = "";

    try {

        const response = await fetch(
            "/analyze",
            {
                method: "POST",
                body: form
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Analysis failed"
            );
        }

        document.getElementById("skills").innerText =
            data.skills;

        document.getElementById("report").innerText =
            data.analysis;

        document.getElementById("result").style.display =
            "block";

    } catch(error) {

        document.getElementById("error").innerText =
            error.message;

    } finally {

        document.getElementById("loading").style.display =
            "none";
    }
}

</script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/health")
def health():
    return {"status": "healthy"}


# ============================================================
# UPLOAD + ANALYZE PROFILE
# ============================================================

@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    role: str = Form(...),
    github_username: str = Form(...)
):

    if not resume.filename.lower().endswith(".pdf"):
        return {
            "error": "Only PDF resumes are supported."
        }

    try:

        pdf_bytes = await resume.read()

        resume_text = extract_pdf(
            pdf_bytes
        )

    except Exception as e:

        return {
            "error": f"Resume processing failed: {str(e)}"
        }

    # Extract skills

    skills_prompt = f"""
Extract ONLY technical skills from this resume.

Include:
- Programming languages
- Frameworks
- Libraries
- Databases
- Cloud
- AI/ML technologies
- Developer tools
- APIs

Do NOT include:
- Soft skills
- Education
- Names
- Companies
- Job titles

Return ONLY a comma-separated list.

Resume:
{resume_text[:9000]}
"""

    try:

        skills_response = llm.invoke(
            skills_prompt
        )

        skills = text_from_response(
            skills_response
        )

    except Exception as e:

        return {
            "error": f"Skill extraction failed: {str(e)}"
        }

    # GitHub

    github_data = get_github(
        github_username.strip()
    )

    # Save profile

    profile["resume"] = resume_text
    profile["skills"] = skills
    profile["role"] = role.strip()
    profile["github"] = github_username.strip()
    profile["github_data"] = github_data

    # Full analysis

    analysis = analyze_profile()

    return {
        "message": "Profile analyzed successfully.",
        "target_role": role,
        "skills": skills,
        "github": github_data,
        "analysis": analysis
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
async def chat(question: str = Form(...)):

    return {
        "answer": answer_question(question)
    }


# ============================================================
# LANGSERVE PLAYGROUND
# ============================================================

def playground_input(text: str):

    return answer_question(text)


playground = RunnableLambda(
    playground_input
)

add_routes(
    app,
    playground,
    path="/agent"
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )
