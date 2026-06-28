import json
from google import genai
from google.genai import types
from app.config import settings

# single client instance
client = genai.Client(api_key=settings.GEMINI_API_KEY)


def generate_explanation(
    job_title: str,
    job_description: str,
    required_skills: list[str],
    candidate_name: str,
    candidate_profile: str,
    candidate_skills: list[str],
    matched_skills: list[str],
    gaps: list[str],
) -> dict:
    """
    Calls Gemini to generate a structured explanation for why
    a candidate is a good or bad fit for a job.

    Returns a dict with keys:
        matched_skills  → list[str]
        gaps            → list[str]
        justification   → str
    """

    prompt = f"""
You are an expert technical recruiter evaluating candidate fit for a job.

JOB DETAILS:
Title: {job_title}
Description: {job_description}
Required Skills: {", ".join(required_skills)}

CANDIDATE DETAILS:
Name: {candidate_name}
Profile: {candidate_profile}
Skills: {", ".join(candidate_skills or [])}
Already matched skills (from automated system): {", ".join(matched_skills)}
Identified gaps (from automated system): {", ".join(gaps)}

Your task:
- Review the candidate profile against the job requirements
- Confirm or refine the matched skills and gaps
- Write a concise 2-3 sentence justification explaining why this candidate
  is or isn't a strong fit

Respond ONLY with a JSON object. No markdown, no extra text, no backticks.
Exact format:
{{
    "matched_skills": ["skill1", "skill2"],
    "gaps": ["gap1", "gap2"],
    "justification": "2-3 sentence explanation here"
}}
""".strip()

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,        # low temp — we want consistent structured output
            max_output_tokens=1000,
        ),
    )

    raw_text = response.text.strip()

    # strip markdown code fences if Gemini adds them despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    parsed = json.loads(raw_text)

    # validate expected keys exist
    return {
        "matched_skills": parsed.get("matched_skills", matched_skills),
        "gaps": parsed.get("gaps", gaps),
        "justification": parsed.get("justification", ""),
    }