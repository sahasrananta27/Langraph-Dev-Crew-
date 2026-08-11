import os
import tempfile
import requests
import uvicorn

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse

from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pypdf import PdfReader


GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API")
)

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not configured.")


MODEL_NAME = "gemini-3.1-flash-lite-preview"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# ============================================================

# APP

# ============================================================

app = FastAPI(
title="Job-Ready Career Assistant",
version="1.0"
)

# ============================================================

# PROFILE MEMORY

# ============================================================

profile = {
"resume": "",
"skills": "",
"role": "",
"github": "",
"github_data": ""
}

# ============================================================

# SAFE GEMINI TEXT

# ============================================================

def text(content):
"""Convert Gemini content into normal string."""

```
if isinstance(content, str):
    return content.strip()

if isinstance(content, list):
    parts = []

    for item in content:
        if isinstance(item, str):
            parts.append(item)

        elif isinstance(item, dict):
            value = item.get("text")

            if value:
                parts.append(str(value))

    return "\n".join(parts).strip()

return str(content).strip()
```

# ============================================================

# PDF TEXT

# ============================================================

def read_pdf(data: bytes) -> str:

```
if not data:
    raise ValueError("PDF is empty.")

path = None

try:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as f:

        f.write(data)
        path = f.name

    reader = PdfReader(path)

    result = []

    for page in reader.pages:
        result.append(page.extract_text() or "")

    resume = "\n".join(result).strip()

    if not resume:
        raise ValueError(
            "Could not extract text. "
            "The PDF may be scanned/image based."
        )

    return resume[:12000]

finally:

    if path and os.path.exists(path):
        os.unlink(path)
```

# ============================================================

# GITHUB

# ============================================================

def github(username: str) -> str:

```
headers = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "Job-Ready-Career-Assistant"
}

try:

    user = requests.get(
        f"https://api.github.com/users/{username}",
        headers=headers,
        timeout=10
    )

    if user.status_code != 200:
        return "GitHub profile not found."

    data = user.json()

    repos = requests.get(
        f"https://api.github.com/users/{username}/repos",
        headers=headers,
        params={
            "sort": "updated",
            "per_page": 10
        },
        timeout=10
    )

    repo_data = []

    if repos.status_code == 200:

        for r in repos.json():

            repo_data.append(
                f"{r.get('name')} | "
                f"{r.get('language') or 'Unknown'} | "
                f"{r.get('description') or 'No description'}"
            )

    return (
        f"Username: {username}\n"
        f"Public repos: {data.get('public_repos', 0)}\n"
        f"Followers: {data.get('followers', 0)}\n"
        f"Bio: {data.get('bio') or 'None'}\n"
        f"Repositories:\n"
        + "\n".join(repo_data)
    )

except Exception:
    return "GitHub information could not be fetched."
```

# ============================================================

# CAREER ANALYSIS

# ============================================================

def analyze_career(question="Give a complete career analysis."):

```
prompt = f"""
```

You are an expert career and placement assistant.

Analyze THIS candidate using the information below.

TARGET ROLE:
{profile["role"]}

RESUME:
{profile["resume"][:7000]}

TECHNICAL SKILLS:
{profile["skills"]}

GITHUB:
{profile["github_data"][:5000]}

USER QUESTION:
{question}

Your job is to give personalized career guidance.

You MUST:

1. Assess whether the candidate can realistically target
   the requested role.

2. Clearly separate:

   * Skills they already have
   * Skills they are missing

3. Identify the most important skill gaps.

4. Tell the candidate whether they should:

   * Apply now
   * Learn more first
   * Apply while learning

5. Suggest other roles they can currently apply for
   based ONLY on their actual skills.

6. If the user asks for a roadmap, provide a practical
   ordered roadmap from their CURRENT level.

7. If useful, recommend projects that improve their profile.

8. Analyze GitHub projects when available.

9. Never invent skills, projects, education or experience.

10. Be honest rather than overly positive.

Use these sections when appropriate:

### Target Role Assessment

### Current Skills

### Skill Gaps

### Other Suitable Roles

### GitHub Analysis

### Recommended Projects

### Roadmap

### Next Steps

Keep the response clear and practical.
"""

```
response = llm.invoke(prompt)

return text(response.content)
```

# ============================================================

# REQUEST MODEL

# ============================================================

class ChatRequest(BaseModel):
question: str

# ============================================================

# ANALYZE RESUME

# ============================================================

@app.post("/analyze")
async def analyze(
resume: UploadFile = File(...),
role: str = Form(...),
github_username: str = Form(...)
):

```
if not resume.filename:
    raise HTTPException(400, "Resume is required.")

if not resume.filename.lower().endswith(".pdf"):
    raise HTTPException(400, "Only PDF files are supported.")

if not role.strip():
    raise HTTPException(400, "Target role is required.")

if not github_username.strip():
    raise HTTPException(400, "GitHub username is required.")

# Read resume
try:

    data = await resume.read()

    resume_text = read_pdf(data)

except Exception as e:

    raise HTTPException(
        400,
        f"Resume processing failed: {e}"
    )

# Extract skills
try:

    prompt = f"""
```

Extract ONLY technical skills from this resume.

Include:
programming languages, frameworks, libraries,
databases, cloud, AI/ML, APIs, developer tools
and platforms.

Do NOT include:
soft skills, education, names, companies,
job titles or personal information.

Return ONLY a comma-separated list.

Resume:
{resume_text[:8000]}
"""

```
    response = llm.invoke(prompt)

    skills = text(response.content)

    if not skills:
        skills = "No technical skills detected."

except Exception as e:

    raise HTTPException(
        500,
        f"Skill extraction failed: {e}"
    )

# GitHub
github_data = github(
    github_username.strip()
)

# Save profile
profile["resume"] = resume_text
profile["skills"] = skills
profile["role"] = role.strip()
profile["github"] = github_username.strip()
profile["github_data"] = github_data

# Generate analysis
report = analyze_career()

return {
    "message": "Profile analyzed successfully.",
    "target_role": profile["role"],
    "skills": skills,
    "github": github_data,
    "report": report
}
```

# ============================================================

# CHAT

# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):

```
if not profile["skills"]:

    return {
        "answer":
        "Please upload your resume, target role "
        "and GitHub username first."
    }

return {
    "answer": analyze_career(req.question)
}
```

# ============================================================

# LANGSERVE PLAYGROUND

# ============================================================

def playground(text_input: str):

```
return analyze_career(text_input)
```

agent = RunnableLambda(playground)

add_routes(
app,
agent,
path="/agent"
)

# ============================================================

# SIMPLE UI

# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Job-Ready Career Assistant</title>

<meta name="viewport"
   content="width=device-width, initial-scale=1">

<style>

body {
    font-family: Arial;
    background: #f1f5f9;
    margin: 0;
}

.container {
    max-width: 700px;
    margin: 40px auto;
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 5px 25px #0002;
}

h1 {
    text-align: center;
}

p {
    color: #64748b;
    text-align: center;
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
    box-sizing: border-box;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
}

button {
    width: 100%;
    padding: 14px;
    margin-top: 22px;
    border: 0;
    border-radius: 8px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:disabled {
    background: #94a3b8;
}

#loading {
    display: none;
    text-align: center;
    margin-top: 20px;
}

#result {
    display: none;
    margin-top: 25px;
    padding: 20px;
    background: #f8fafc;
    border-radius: 10px;
}

#report {
    white-space: pre-wrap;
    line-height: 1.6;
}

#error {
    color: #dc2626;
    margin-top: 15px;
}

</style>

</head>

<body>

<div class="container">

<h1>🎯 Job-Ready Career Assistant</h1>

<p>
Upload your resume and discover your career readiness.
</p>

<label>📄 Resume PDF</label>

<input
type="file"
id="resume"
accept=".pdf"

>

<label>💼 Target Role</label>

<input
id="role"
placeholder="Example: Python Developer"

>

<label>🐙 GitHub Username</label>

<input
id="github"
placeholder="Example: sahasra123"

>

<button id="button" onclick="analyze()">
Analyze My Career
</button>

<div id="loading">
⏳ Analyzing resume, skills and GitHub...
</div>

<div id="error"></div>

<div id="result">

<h3>🧠 Detected Skills</h3>

<div id="skills"></div>

<h3>📊 Career Analysis</h3>

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

    const error =
        document.getElementById("error");

    const result =
        document.getElementById("result");

    const loading =
        document.getElementById("loading");

    const button =
        document.getElementById("button");

    error.innerText = "";

    if (!resume)
        return error.innerText =
            "Please upload your resume.";

    if (!role)
        return error.innerText =
            "Please enter your target role.";

    if (!github)
        return error.innerText =
            "Please enter your GitHub username.";

    const form = new FormData();

    form.append("resume", resume);
    form.append("role", role);
    form.append("github_username", github);

    button.disabled = true;
    loading.style.display = "block";
    result.style.display = "none";

    try {

        const response = await fetch(
            "/analyze",
            {
                method: "POST",
                body: form
            }
        );

        const data = await response.json();

        if (!response.ok)
            throw new Error(
                data.detail || "Analysis failed."
            );

        document.getElementById("skills")
            .innerText = data.skills;

        document.getElementById("report")
            .innerText = data.report;

        result.style.display = "block";

    } catch (e) {

        error.innerText = e.message;

    } finally {

        button.disabled = false;
        loading.style.display = "none";

    }
}

</script>

</body>
</html>
"""

# ============================================================

# HOME

# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():
return HTML

# ============================================================

# HEALTH

# ============================================================

@app.get("/health")
def health():
return {"status": "healthy"}

# ============================================================

# SERVER

# ============================================================

if **name** == "**main**":

```
uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=int(os.getenv("PORT", 8000))
)
```
