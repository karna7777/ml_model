"""
llm_client.py
─────────────
LLM client module for the Credit Card Approval Lending Decision Support system.

Uses the Groq API (free tier) with Llama-3.1-8b-instant to generate structured
credit assessment reports.  Includes a rule-based fallback when the API key is
missing or the call fails.
"""

import json
import re
from typing import Optional

# ────────────────────────────────────────────────────────────────
# Groq configuration
# ────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

DISCLAIMER_TEXT = (
    "This AI-generated report is advisory only. Final lending decisions must be "
    "made by a qualified human officer in compliance with applicable RBI "
    "regulations. This system does not discriminate on the basis of gender, "
    "religion, caste, or any other protected characteristic."
)

EDUCATION_DISPLAY = {
    "Secondary / secondary special": "secondary education",
    "Higher education": "higher education",
    "Incomplete higher": "incomplete higher education",
    "Lower secondary": "lower secondary education",
    "Academic degree": "an academic degree",
}

HOUSING_DISPLAY = {
    "House / apartment": "owns a house or apartment",
    "With parents": "lives with family",
    "Rented apartment": "lives in rented accommodation",
    "Municipal apartment": "lives in municipal housing",
    "Co-op apartment": "lives in cooperative housing",
    "Office apartment": "lives in employer-provided housing",
}


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _clean_driver(driver: str) -> str:
    pretty = str(driver).replace("_", " ").lower()
    replacements = {
        "amt income total": "income level",
        "employed years": "employment tenure",
        "name education type": "education profile",
        "name family status": "family profile",
        "name housing type": "housing profile",
        "flag own realty": "property ownership",
        "flag own car": "vehicle ownership",
        "cnt children": "dependent obligations",
        "code gender": "demographic profile",
        "job": "occupation profile",
        "age": "age profile",
    }
    return replacements.get(pretty, pretty)


def _risk_band_text(risk_score: float, risk_class: str) -> str:
    return f"{risk_class} ({risk_score:.1%} estimated default probability)"


def _profile_snapshot(borrower_profile: dict) -> str:
    age = int(float(borrower_profile.get("AGE", 0))) if borrower_profile.get("AGE") else None
    employed_years = float(borrower_profile.get("EMPLOYED_YEARS", 0) or 0)
    income = borrower_profile.get("AMT_INCOME_TOTAL")
    education = EDUCATION_DISPLAY.get(
        borrower_profile.get("NAME_EDUCATION_TYPE"),
        str(borrower_profile.get("NAME_EDUCATION_TYPE", "their reported education")),
    )
    housing = HOUSING_DISPLAY.get(
        borrower_profile.get("NAME_HOUSING_TYPE"),
        str(borrower_profile.get("NAME_HOUSING_TYPE", "their current housing arrangement")).lower(),
    )
    children = borrower_profile.get("CNT_CHILDREN", "No children")
    job = str(borrower_profile.get("JOB", "the stated occupation")).strip()
    job_phrase = (
        f"reported occupation category '{job}'"
        if job.lower() not in {"n/a", "the stated occupation"}
        else "has a stated occupation"
    )

    parts = []
    if age:
        parts.append(f"{age}-year-old applicant")
    if isinstance(income, (int, float)):
        parts.append(f"annual income of INR {income:,.0f}")
    parts.append(job_phrase)
    parts.append(f"has {education}")
    parts.append(housing)
    parts.append(f"reported employment tenure of {employed_years:.1f} years")
    parts.append(family_phrase(children))
    return ", ".join(parts)


def family_phrase(children: str) -> str:
    if str(children) == "No children":
        return "no dependent children"
    if str(children) == "1 children":
        return "one dependent child"
    return "multiple dependent children"


def _recommendation_from_risk(risk_class: str, risk_score: float) -> tuple[str, str]:
    if risk_class == "High Risk" or risk_score >= 0.60:
        return (
            "Reject",
            "Do not proceed under standard unsecured lending policy. Recommend manual underwriting only if "
            "additional collateral, strong co-applicant support, or updated income proof materially reduces risk.",
        )
    if risk_class == "Medium Risk" or risk_score >= 0.30:
        return (
            "Conditional Approval",
            "Proceed only with mitigants such as a lower credit limit, stricter income verification, "
            "short review cycle, or additional supporting documentation before final approval.",
        )
    return (
        "Approve",
        "Proceed under standard policy controls, with onboarding checks and credit exposure aligned to "
        "the applicant's verified income profile.",
    )


def _recommendation_narrative(recommendation: str) -> str:
    mapping = {
        "Approve": "approval",
        "Conditional Approval": "conditional approval",
        "Reject": "rejection",
    }
    return mapping.get(recommendation, recommendation.lower())


def _regulation_highlights(regulations: list[str]) -> list[str]:
    highlights = []
    for item in regulations[:2]:
        text = str(item).strip()
        if not text:
            continue
        first_sentence = text.split(". ")[0].strip()
        highlights.append(first_sentence if first_sentence.endswith(".") else first_sentence + ".")
    return highlights

def _build_prompt(
    borrower_profile: dict,
    risk_score: float,
    risk_class: str,
    risk_drivers: list,
    regulations: list,
) -> str:
    """Build the system + user prompt for the LLM."""

    profile_name = borrower_profile.get("NAME", "Applicant")
    profile_str = "\n".join(f"  - {k}: {v}" for k, v in borrower_profile.items())
    drivers_str = ", ".join(risk_drivers) if risk_drivers else "N/A"
    regs_str = "\n".join(f"  [{i+1}] {r}" for i, r in enumerate(regulations))

    prompt = f"""You are an expert credit risk analyst AI preparing a polished lending decision-support note for an academic demonstration of an intelligent credit underwriting system. The output must sound professional, evidence-based, and concise.

BORROWER PROFILE:
{profile_str}

ML RISK ASSESSMENT:
  - Risk Score (probability of default): {risk_score:.4f}
  - Risk Class: {risk_class}
  - Top Risk Drivers: {drivers_str}

RELEVANT REGULATIONS:
{regs_str}

INSTRUCTIONS:
1. Produce ONLY valid JSON — no markdown fences, no commentary.
2. Use this exact schema:
{{
  "borrower_summary": "2-3 sentences. Start with the applicant name and state the decision in a formal credit-review tone. Summarize the most relevant borrower facts only.",
  "risk_analysis": "3-4 sentences explaining the score, risk class, and key drivers in professional underwriting language",
  "lending_recommendation": "Approve / Conditional Approval / Reject",
  "recommended_action": "Specific next steps for the lending officer with concrete risk mitigations or approval path",
  "regulatory_references": ["brief regulation summary 1", "brief regulation summary 2"],
  "disclaimer": "{DISCLAIMER_TEXT}"
}}
3. Do not mention protected traits as reasons for the decision.
4. Keep the reasoning consistent with the score and risk class.
5. Do not exaggerate; write like a disciplined analyst.

Return ONLY the JSON object. Nothing else."""

    return prompt


def _extract_json_from_text(raw: str) -> Optional[dict]:
    """Try hard to pull a JSON object out of potentially dirty LLM output."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    cleaned = cleaned.strip("`").strip()

    # Attempt direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find the first { … } block
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _fallback_from_text(raw: str, risk_class: str, regulations: list) -> dict:
    """Build a best-effort report dict from raw text when JSON parsing fails."""
    return {
        "borrower_summary": raw[:300] if raw else "Unable to generate summary.",
        "risk_analysis": raw[300:700] if len(raw) > 300 else "See borrower summary.",
        "lending_recommendation": (
            "Reject" if risk_class == "High Risk"
            else ("Conditional Approval" if risk_class == "Medium Risk" else "Approve")
        ),
        "recommended_action": "Manual review recommended due to AI report generation issue.",
        "regulatory_references": regulations[:2] if regulations else [],
        "disclaimer": DISCLAIMER_TEXT,
    }


def _validate_report(report: dict, borrower_profile: dict, risk_score: float, risk_class: str,
                     risk_drivers: list, regulations: list) -> dict:
    """Ensure the final report is structurally complete and consistent."""
    fallback = _rule_based_report(
        borrower_profile, risk_score, risk_class, risk_drivers, regulations
    )

    if not isinstance(report, dict):
        return fallback

    validated = dict(fallback)
    for key, value in report.items():
        if value not in (None, "", []):
            validated[key] = value

    rec = str(validated.get("lending_recommendation", "")).strip().lower()
    if "reject" in rec:
        validated["lending_recommendation"] = "Reject"
    elif "conditional" in rec:
        validated["lending_recommendation"] = "Conditional Approval"
    elif "approve" in rec:
        validated["lending_recommendation"] = "Approve"
    else:
        validated["lending_recommendation"] = fallback["lending_recommendation"]

    if not isinstance(validated.get("regulatory_references"), list):
        validated["regulatory_references"] = fallback["regulatory_references"]

    validated["disclaimer"] = DISCLAIMER_TEXT
    return validated


def _rule_based_report(
    borrower_profile: dict,
    risk_score: float,
    risk_class: str,
    risk_drivers: list,
    regulations: list,
) -> dict:
    """Pure-Python rule-based fallback when the Groq API is unavailable."""

    recommendation, action = _recommendation_from_risk(risk_class, risk_score)
    applicant_name = borrower_profile.get("NAME", "Applicant")
    snapshot = _profile_snapshot(borrower_profile)
    cleaned_drivers = [_clean_driver(driver) for driver in risk_drivers[:3] if str(driver).strip()]
    driver_text = ", ".join(cleaned_drivers) if cleaned_drivers else "income stability and repayment capacity indicators"

    summary = (
        f"{applicant_name}'s application is recommended for {_recommendation_narrative(recommendation)} under the current decision-support workflow. "
        f"The profile reflects a {snapshot}. "
        f"Based on the trained model output, the case falls in the {_risk_band_text(risk_score, risk_class)} band."
    )

    analysis = (
        f"The model estimates default probability at {risk_score:.2%}, which maps to the {risk_class} category used in this system. "
        f"The strongest contributing drivers in the present assessment are {driver_text}. "
        f"This indicates that the recommendation should be guided by verified repayment capacity, employment continuity, and overall exposure discipline."
    )
    if risk_score >= 0.60:
        analysis += " The score is sufficiently elevated to justify a conservative stance and a non-standard approval path."
    elif risk_score >= 0.30:
        analysis += " The score is not severe, but it is high enough to justify conditional approval rather than immediate standard acceptance."
    else:
        analysis += " The score remains within a comparatively comfortable range, supporting a standard approval view subject to normal checks."

    return {
        "borrower_summary": summary,
        "risk_analysis": analysis,
        "lending_recommendation": recommendation,
        "recommended_action": action,
        "regulatory_references": _regulation_highlights(regulations),
        "disclaimer": DISCLAIMER_TEXT,
    }


# ────────────────────────────────────────────────────────────────
# Main public function
# ────────────────────────────────────────────────────────────────

def generate_credit_report(
    borrower_profile: dict,
    risk_score: float,
    risk_class: str,
    risk_drivers: list,
    regulations: list,
) -> dict:
    """
    Generate a structured credit assessment report.

    Tries the Groq API first; falls back to rule-based generation if the API
    key is missing or the call fails.

    Returns
    -------
    dict  with keys: borrower_summary, risk_analysis, lending_recommendation,
          recommended_action, regulatory_references, disclaimer
    """

    # ── Guard: no API key → rule-based fallback ──
    if not GROQ_API_KEY or GROQ_API_KEY.strip() == "":
        return _rule_based_report(
            borrower_profile, risk_score, risk_class, risk_drivers, regulations
        )

    # ── Try Groq API ──
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        prompt = _build_prompt(
            borrower_profile, risk_score, risk_class, risk_drivers, regulations
        )

        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a credit risk analyst AI. Always respond with valid "
                        "JSON only. No markdown, no explanation outside the JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        raw_text = chat_completion.choices[0].message.content.strip()

        # Try to parse JSON
        parsed = _extract_json_from_text(raw_text)
        if parsed is not None:
            return _validate_report(
                parsed, borrower_profile, risk_score, risk_class, risk_drivers, regulations
            )

        # JSON parsing failed — build from raw text
        return _validate_report(
            _fallback_from_text(raw_text, risk_class, regulations),
            borrower_profile,
            risk_score,
            risk_class,
            risk_drivers,
            regulations,
        )

    except Exception as e:
        # API call failed entirely — use rule-based fallback
        print(f"[llm_client] Groq API error: {e}. Using rule-based fallback.")
        return _validate_report(
            _rule_based_report(
                borrower_profile, risk_score, risk_class, risk_drivers, regulations
            ),
            borrower_profile,
            risk_score,
            risk_class,
            risk_drivers,
            regulations,
        )


# ────────────────────────────────────────────────────────────────
# Quick self-test
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_profile = {
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": "No children",
        "AMT_INCOME_TOTAL": 300000,
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "JOB": "Managers",
        "AGE": 35.0,
        "EMPLOYED_YEARS": 8.0,
    }

    report = generate_credit_report(
        borrower_profile=sample_profile,
        risk_score=0.15,
        risk_class="Low Risk",
        risk_drivers=["AMT_INCOME_TOTAL", "EMPLOYED_YEARS", "AGE"],
        regulations=[
            "Income-to-Loan Ratio Limits: EMI should not exceed 40% of net monthly income.",
            "Employment Stability: Minimum 2 years continuous employment preferred.",
        ],
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
