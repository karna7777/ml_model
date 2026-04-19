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


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

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

    prompt = f"""You are an expert credit risk analyst AI. Given the following borrower data, ML risk assessment, and relevant lending regulations, produce a structured lending decision report.

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
  "borrower_summary": "Start with 'Dear {profile_name}, your application has been ACCEPTED (or REJECTED/CONDITIONALLY APPROVED).' Then give a 2-3 sentence summary of the borrower profile.",
  "risk_analysis": "3-4 sentences explaining the risk score referencing specific features and drivers",
  "lending_recommendation": "Approve / Conditional Approval / Reject",
  "recommended_action": "Specific actionable advice for the lending officer",
  "regulatory_references": ["regulation text 1", "regulation text 2"],
  "disclaimer": "{DISCLAIMER_TEXT}"
}}

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


def _rule_based_report(
    borrower_profile: dict,
    risk_score: float,
    risk_class: str,
    risk_drivers: list,
    regulations: list,
) -> dict:
    """Pure-Python rule-based fallback when the Groq API is unavailable."""

    # Determine recommendation
    if risk_class == "High Risk":
        recommendation = "Reject"
        action = (
            "Application should be declined. Key risk factors include: "
            + ", ".join(risk_drivers[:3])
            + ". Advise the applicant to improve these areas before re-applying."
        )
    elif risk_class == "Medium Risk":
        recommendation = "Conditional Approval"
        action = (
            "Application may be approved with conditions. Recommend: "
            "lower credit limit, mandatory co-applicant or collateral, "
            "and enhanced monitoring for the first 12 months."
        )
    else:
        recommendation = "Approve"
        action = (
            "Application meets standard lending criteria. Recommend standard credit "
            "limit based on income tier and proceed with standard onboarding."
        )

    # Build summary from profile
    gender = borrower_profile.get("CODE_GENDER", "N/A")
    income = borrower_profile.get("AMT_INCOME_TOTAL", "N/A")
    age = borrower_profile.get("AGE", "N/A")
    job = borrower_profile.get("JOB", "N/A")

    summary = (
        f"The applicant is a {age}-year-old {gender} with an annual income of "
        f"₹{income:,.0f}. " if isinstance(income, (int, float)) else
        f"The applicant (Gender: {gender}, Age: {age}) works as {job}. "
    )
    summary += f"They are classified as {risk_class} with a default probability of {risk_score:.2%}."

    analysis = (
        f"The ML model assigns a default probability of {risk_score:.4f}, placing the applicant "
        f"in the '{risk_class}' category. "
        f"The primary risk drivers are: {', '.join(risk_drivers[:3])}. "
    )
    if risk_score > 0.6:
        analysis += "The elevated risk score suggests significant concern about repayment capacity."
    elif risk_score > 0.3:
        analysis += "The moderate risk score suggests some areas of concern that warrant additional review."
    else:
        analysis += "The low risk score indicates strong creditworthiness indicators."

    return {
        "borrower_summary": summary,
        "risk_analysis": analysis,
        "lending_recommendation": recommendation,
        "recommended_action": action,
        "regulatory_references": regulations[:2] if regulations else [],
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
            # Ensure disclaimer is always present and correct
            parsed["disclaimer"] = DISCLAIMER_TEXT
            return parsed

        # JSON parsing failed — build from raw text
        return _fallback_from_text(raw_text, risk_class, regulations)

    except Exception as e:
        # API call failed entirely — use rule-based fallback
        print(f"[llm_client] Groq API error: {e}. Using rule-based fallback.")
        return _rule_based_report(
            borrower_profile, risk_score, risk_class, risk_drivers, regulations
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
