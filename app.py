
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
        parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        return "\n".join(parts).strip()

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
        ).strip()

        if not text:
            raise ValueError(
                "Could not extract text from PDF. "
                "The resume may be scanned or image-based."
            )

        return text[:12000]

    finally:
        if os.path.exists(path):
            os.unlink(path)


# ============================================================
# ROLE VALIDATION
# ============================================================

VALID_ROLE_KEYWORDS = [
    "software",
    "developer",
    "engineer",
    "programmer",
    "data analyst",
    "data scientist",
    "machine learning",
    "ml engineer",
    "ai engineer",
    "ai developer",
    "backend",
    "frontend",
    "full stack",
    "fullstack",
    "web developer",
    "mobile developer",
    "android developer",
    "ios developer",
    "java developer",
    "python developer",
    "cloud engineer",
    "devops",
    "cybersecurity",
    "security analyst",
    "database",
    "sql developer",
    "business analyst",
    "qa engineer",
    "test engineer",
    "automation engineer",
    "data engineer",
    "network engineer",
    "technical support"
]


def valid_role(role):
    role = role.lower().strip()

    if len(role) < 3:
        return False

    return any(
        keyword in role
        for keyword in VALID_ROLE_KEYWORDS
    )


# ============================================================
# GITHUB
# ============================================================

def get_github(username):

    username = username.strip()

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "PlacementGuider"
    }

    token = os.getenv("GITHUB_TOKEN")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:

        user_response = requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers,
            timeout=10
        )

        if user_response.status_code != 200:
            return "GitHub profile not found."

        user = user_response.json()

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

        repo_text = []

        for repo in repos:

            language = repo.get("language")

            if language:
                languages[language] = (
                    languages.get(language, 0) + 1
                )

            repo_text.append(
                f"- {repo.get('name')}: "
                f"{repo.get('description') or 'No description'} "
                f"({language or 'Unknown'})"
            )

        return f"""
GitHub username: {username}
Public repositories: {user.get('public_repos', 0)}
Followers: {user.get('followers', 0)}
Bio: {user.get('bio') or 'Not provided'}
Top languages: {languages}

Recent repositories:
{chr(10).join(repo_text) if repo_text else "No public repositories found."}
""".strip()

    except Exception:
        return "GitHub analysis unavailable."


# ============================================================
# INITIAL PROFILE ANALYSIS
# ============================================================

def analyze_profile():

    prompt = f"""
You are PlacementGuider, a professional career advisor.

Analyze this candidate only using the provided information.

TARGET ROLE:
{profile["role"]}

RESUME:
{profile["resume"][:9000]}

CURRENT TECHNICAL SKILLS:
{profile["skills"]}

GITHUB:
{profile["github_data"]}

Give a practical and honest career assessment.

Use EXACTLY these sections:

## 1. Target Role Readiness
Give one verdict:
- Ready
- Almost Ready
- Needs More Preparation

Explain why.

Also clearly answer:
"Can this candidate achieve the target role?"

## 2. Current Skills
List the strongest technical skills relevant to the target role.

## 3. Skill Gaps
List the most important missing skills.
Prioritize them from HIGH to LOW.

## 4. Other Suitable Roles
Suggest 3-5 realistic roles based ONLY on the
candidate's current skills.

## 5. Learning Roadmap
Give an ordered roadmap:

Phase 1 - Fundamentals
Phase 2 - Role-specific skills
Phase 3 - Projects
Phase 4 - Interview preparation

## 6. Project Recommendations
Suggest 2-3 projects that would strengthen
the candidate for the target role.

## 7. GitHub Improvements
Give practical GitHub improvements.

## 8. Resume Improvements
Give specific improvements for:
- Skills
- Projects
- Resume structure
- Keywords
- Achievement descriptions

IMPORTANT:
- Do not invent skills.
- Do not invent experience.
- Do not invent projects.
- Do not assume an invalid target role means Software Engineer.
- Judge the exact target role provided.
- Be honest about skill gaps.
- Keep the answer practical.
"""

    return text_from_response(
        llm.invoke(prompt)
    )


# ============================================================
# CAREER QUESTION VALIDATION
# ============================================================

CAREER_KEYWORDS = [
    "career",
    "job",
    "role",
    "placement",
    "resume",
    "cv",
    "skill",
    "learn",
    "learning",
    "roadmap",
    "project",
    "github",
    "interview",
    "salary",
    "hiring",
    "developer",
    "engineer",
    "programmer",
    "software",
    "python",
    "java",
    "sql",
    "frontend",
    "backend",
    "full stack",
    "machine learning",
    "ai",
    "data analyst",
    "data scientist",
    "devops",
    "cloud",
    "certification",
    "internship"
]


def is_career_question(question):

    question = question.lower().strip()

    return any(
        keyword in question
        for keyword in CAREER_KEYWORDS
    )


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(question):

    if not profile["resume"]:
        return (
            "Please analyze your resume first. "
            "Upload your resume, target role and GitHub username."
        )

    question = question.strip()

    if not question:
        return "Please enter a career-related question."

    if not is_career_question(question):
        return (
            "Sorry, I can only answer career-related "
            "questions based on your resume, target role, "
            "skills and GitHub profile."
        )

    prompt = f"""
You are PlacementGuider, a personalized career assistant.

Candidate profile:

TARGET ROLE:
{profile["role"]}

RESUME:
{profile["resume"][:8000]}

CURRENT SKILLS:
{profile["skills"]}

GITHUB:
{profile["github_data"]}

USER QUESTION:
{question}

Rules:

1. Answer ONLY the career question.

2. Personalize the answer using the candidate profile.

3. Do not invent skills, education, experience,
projects, certifications or achievements.

4. If the candidate asks whether they can achieve
the target role, compare their current skills
with the target role and give an honest verdict.

5. If they ask what skills to learn, prioritize
the missing skills.

6. If they ask for a roadmap, give an ordered roadmap.

7. If they ask about resume improvement, give
specific resume improvements.

8. If they ask about GitHub, use the provided
GitHub information.

9. If they ask about other jobs, recommend roles
that realistically match their current profile.

10. Do not answer jokes, romance, entertainment,
general trivia, politics, weather or unrelated questions.

Keep the response practical and concise.
"""

    try:

        return text_from_response(
            llm.invoke(prompt)
        )

    except Exception:
        return (
            "Sorry, I couldn't analyze that career "
            "question right now. Please try again."
        )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PlacementGuider Career Assistant",
    version="3.0",
    description="Personalized resume and career analysis"
)


# ============================================================
# HOME UI
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>PlacementGuider</title>

<meta
name="viewport"
content="width=device-width, initial-scale=1"
>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f1f5f9;
    margin: 0;
}

.container {
    max-width: 850px;
    margin: 35px auto;
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

input,
textarea {
    width: 100%;
    padding: 12px;
    margin-top: 7px;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    box-sizing: border-box;
    font-size: 15px;
}

button {
    width: 100%;
    padding: 13px;
    margin-top: 20px;
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

button:disabled {
    background: #94a3b8;
}

#loading {
    display: none;
    text-align: center;
    margin-top: 20px;
    color: #2563eb;
}

#result,
#questionBox {
    display: none;
    margin-top: 30px;
}

.card {
    background: #eff6ff;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}

#skills {
    line-height: 1.7;
}

#report,
#answer {
    white-space: pre-wrap;
    line-height: 1.7;
}

.error {
    color: #dc2626;
    margin-top: 15px;
}

.success {
    color: #16a34a;
    margin-top: 15px;
}

</style>

</head>

<body>

<div class="container">

<h1>🎯 PlacementGuider</h1>

<p class="subtitle">
Personalized Resume & Career Assistant
</p>


<label>📄 Resume PDF</label>

<input
type="file"
id="resume"
accept=".pdf"
>


<label>💼 Target Career Role</label>

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


<button
id="analyzeButton"
onclick="analyzeProfile()"
>
Analyze My Profile
</button>


<div id="loading">
⏳ Analyzing resume, skills and GitHub...
</div>


<div id="error" class="error"></div>


<div id="result">

<div class="card">

<h2>🛠 Current Technical Skills</h2>

<div id="skills"></div>

</div>


<div>

<h2>📊 Career Analysis</h2>

<div id="report"></div>

</div>

</div>


<!-- OPTIONAL QUESTION SECTION -->

<div id="questionBox">

<h2>💬 Ask a Career Question</h2>

<p>
This section is optional. You can ask questions
about your career, skills, roadmap, resume,
GitHub, projects or job roles.
</p>

<textarea
id="question"
rows="4"
placeholder="Example: Can I become a Python Developer with my current skills?"
></textarea>

<button
onclick="askQuestion()"
>
Ask Question
</button>

<div id="questionLoading"></div>

<div class="card">

<div id="answer"></div>

</div>

</div>


</div>


<script>

async function analyzeProfile() {

    const resume =
        document.getElementById("resume").files[0];

    const role =
        document.getElementById("role").value.trim();

    const github =
        document.getElementById("github").value.trim();

    const error =
        document.getElementById("error");

    if (!resume) {
        error.innerText =
            "Please upload your resume PDF.";
        return;
    }

    if (!role) {
        error.innerText =
            "Please enter your target career role.";
        return;
    }

    if (!github) {
        error.innerText =
            "Please enter your GitHub username.";
        return;
    }

    const form = new FormData();

    form.append("resume", resume);
    form.append("role", role);
    form.append("github_username", github);

    const button =
        document.getElementById("analyzeButton");

    button.disabled = true;

    document.getElementById("loading")
        .style.display = "block";

    document.getElementById("result")
        .style.display = "none";

    document.getElementById("questionBox")
        .style.display = "none";

    error.innerText = "";

    try {

        const response = await fetch(
            "/analyze",
            {
                method: "POST",
                body: form
            }
        );

        const data = await response.json();

        if (!response.ok || data.error) {

            throw new Error(
                data.error ||
                data.detail ||
                "Profile analysis failed."
            );
        }

        document.getElementById("skills")
            .innerText = data.skills;

        document.getElementById("report")
            .innerText = data.analysis;

        document.getElementById("result")
            .style.display = "block";

        // Question section is OPTIONAL
        document.getElementById("questionBox")
            .style.display = "block";

    } catch (error) {

        document.getElementById("error")
            .innerText = error.message;

    } finally {

        button.disabled = false;

        document.getElementById("loading")
            .style.display = "none";
    }
}


async function askQuestion() {

    const question =
        document.getElementById("question")
        .value.trim();

    if (!question) {

        document.getElementById("answer")
            .innerText =
            "Please enter a career-related question.";

        return;
    }

    const form = new FormData();

    form.append("question", question);

    const loading =
        document.getElementById("questionLoading");

    loading.innerText =
        "⏳ Thinking...";

    try {

        const response = await fetch(
            "/chat",
            {
                method: "POST",
                body: form
            }
        );

        const data = await response.json();

        document.getElementById("answer")
            .innerText = data.answer;

    } catch (error) {

        document.getElementById("answer")
            .innerText =
            "Could not connect to the server.";

    } finally {

        loading.innerText = "";
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
    return {
        "status": "healthy",
        "service": "PlacementGuider"
    }


# ============================================================
# ANALYZE PROFILE
# ============================================================

@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    role: str = Form(...),
    github_username: str = Form(...)
):

    role = role.strip()
    github_username = github_username.strip()

    # --------------------------------------------------------
    # Validate role
    # --------------------------------------------------------

    if not valid_role(role):

        return {
            "error": (
                "Sorry, I can only analyze valid career roles. "
                "Examples: Python Developer, Java Developer, "
                "Data Analyst, ML Engineer, Software Engineer."
            )
        }

    # --------------------------------------------------------
    # Validate PDF
    # --------------------------------------------------------

    if not resume.filename:
        return {
            "error": "Please upload a resume."
        }

    if not resume.filename.lower().endswith(".pdf"):
        return {
            "error": "Only PDF resumes are supported."
        }

    # --------------------------------------------------------
    # Extract resume
    # --------------------------------------------------------

    try:

        pdf_bytes = await resume.read()

        resume_text = extract_pdf(pdf_bytes)

    except Exception as e:

        return {
            "error": f"Resume processing failed: {str(e)}"
        }

    # --------------------------------------------------------
    # Extract skills
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GitHub
    # --------------------------------------------------------

    github_data = get_github(
        github_username
    )

    # --------------------------------------------------------
    # Save profile
    # --------------------------------------------------------

    profile["resume"] = resume_text
    profile["skills"] = skills
    profile["role"] = role
    profile["github"] = github_username
    profile["github_data"] = github_data

    # --------------------------------------------------------
    # Full analysis
    # --------------------------------------------------------

    analysis = analyze_profile()

    return {
        "message": "Profile analyzed successfully.",
        "target_role": role,
        "skills": skills,
        "github": github_data,
        "analysis": analysis
    }


# ============================================================
# OPTIONAL CHAT
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

