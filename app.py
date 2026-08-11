
import os
import tempfile
import requests

from fastapi import FastAPI, UploadFile, File, Form
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pypdf import PdfReader


# -------------------- LLM --------------------

MODEL_NAME = "models/gemini-3.5-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    
)


# -------------------- Tools --------------------

@tool
def job_search(role: str) -> str:
    """Search for current job/internship postings for a role."""
    from langchain_community.tools import DuckDuckGoSearchRun

    raw = DuckDuckGoSearchRun().run(
        f"{role} job openings required skills site:linkedin.com OR site:naukri.com"
    )

    prompt = f"""
Extract from these job search snippets for "{role}":

1. Top technical skills
2. Common qualifications
3. Nice-to-have skills

Data:
{raw}

Return concise bullet points.
"""

    return llm.invoke(prompt).content


@tool
def skill_gap_analysis(resume_skills: str, market_skills: str) -> str:
    """Compare resume skills with market skills."""

    prompt = f"""
STUDENT SKILLS:
{resume_skills}

MARKET SKILLS:
{market_skills}

Return:
1. Matched Skills
2. Critical Gaps
3. Quick Wins
4. Readiness Verdict
"""

    return llm.invoke(prompt).content


@tool
def recommend_projects(skill_gaps: str) -> str:
    """Recommend projects for missing skills."""

    prompt = f"""
Skill gaps:
{skill_gaps}

Suggest exactly 4 projects with:
- Title
- Skills demonstrated
- One-line scope
- Estimated build time
"""

    return llm.invoke(prompt).content


@tool
def github_profile_check(github_username: str) -> str:
    """Analyze a GitHub profile."""

    headers = {}
    token = os.getenv("GITHUB_TOKEN")

    if token:
        headers["Authorization"] = f"token {token}"

    user_res = requests.get(
        f"https://api.github.com/users/{github_username}",
        headers=headers,
    )

    if user_res.status_code != 200:
        return f"Could not fetch GitHub profile for '{github_username}'."

    user = user_res.json()

    repos_res = requests.get(
        f"https://api.github.com/users/{github_username}/repos?sort=pushed&per_page=10",
        headers=headers,
    )

    repos = repos_res.json() if repos_res.status_code == 200 else []

    languages = {}
    recently_active = 0

    for r in repos:
        lang = r.get("language")
        if lang:
            languages[lang] = languages.get(lang, 0) + 1

        if r.get("pushed_at", "") >= "2025-01-01":
            recently_active += 1

    return (
        f"GitHub: {github_username}\n"
        f"Public repos: {user.get('public_repos')}\n"
        f"Followers: {user.get('followers')}\n"
        f"Bio: {user.get('bio') or 'None'}\n"
        f"Top languages: {dict(sorted(languages.items(), key=lambda x: -x[1]))}\n"
        f"Active repos since 2025: {recently_active}/{len(repos)}"
    )


# -------------------- Agent --------------------

# -------------------- Agent --------------------

from langchain_core.runnables import RunnableLambda

SYSTEM_PROMPT = """
You are "PlacementPrep", an AI placement-readiness agent.

Given resume skills, a target role, and a GitHub username:

1. Call job_search on the target role.
2. Call skill_gap_analysis comparing resume skills to market skills.
3. Call recommend_projects using the critical gaps.
4. Call github_profile_check on the GitHub username.
5. Write ONE final report with sections:
   - Market Snapshot
   - Skill Gap
   - Recommended Projects
   - GitHub Health
   - Action Plan (Next 30 Days)

Use clean Markdown formatting with headings, bullet points, and tables where useful.
Do not return tool traces, JSON, or internal messages.
"""

agent = create_agent(
    model=llm,
    tools=[
        job_search,
        skill_gap_analysis,
        recommend_projects,
        github_profile_check,
    ],
    system_prompt=SYSTEM_PROMPT,
)

# Wrapper that returns only the final assistant response
def run_agent(input_data: dict) -> str:
    result = agent.invoke(input_data)

    # Get the final assistant message
    final_message = result["messages"][-1].content

    # Handle list response from Gemini
    if isinstance(final_message, list):
        final_message = " ".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in final_message
        )

    return str(final_message)

# Create a clean runnable
clean_agent = RunnableLambda(run_agent)

# -------------------- FastAPI App --------------------

app = FastAPI(
    title="Placement-Ready AI Agent",
    version="1.0",
    description="LangServe deployment on Render",
)

add_routes(app, clean_agent, path="/agent")


@app.get("/")
def root():
    return {"message": "Placement-Ready AI Agent is running!"}


@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    role: str = Form(...),
    github_username: str = Form(...),
):
    """Upload resume + role + GitHub username."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    text = "".join(p.extract_text() or "" for p in reader.pages)

    os.unlink(tmp_path)

    skills_prompt = f"""
Extract ONLY technical skills as a comma-separated list.

Resume text:
{text[:6000]}
"""

    response = llm.invoke(skills_prompt)
    content = response.content

    if isinstance(content, list):
        content = " ".join(str(x) for x in content)

    resume_skills = str(content).strip()

    user_input = f"""
Student profile:

- Target role: {role}
- Resume skills: {resume_skills}
- GitHub username: {github_username}

Run the full placement-readiness analysis and give me the final report.
"""

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    final_message = result["messages"][-1].content

    return {
        "resume_skills": resume_skills,
        "report": final_message,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    uvicorn.run("app:app", host="0.0.0.0", port=port)
