"""
agent_workflow.py
─────────────────
LangGraph-based agentic workflow for the Credit Card Approval
Lending Decision Support system.

Graph:  input_parser → risk_analyzer → regulation_retriever
          → report_generator → output_formatter
"""

from __future__ import annotations

import traceback
from typing import TypedDict

import numpy as np
import pandas as pd
import streamlit as st
from langgraph.graph import StateGraph, END

from regulations_rag import initialize_rag, retrieve_regulations
from llm_client import generate_credit_report


# ────────────────────────────────────────────────────────────────
# State schema
# ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    borrower_profile: dict
    risk_score: float
    risk_class: str          # "Low Risk" / "Medium Risk" / "High Risk"
    risk_drivers: list       # top 3 feature names driving the prediction
    retrieved_regulations: list
    llm_report: dict
    agent_trace: list        # log of each node's action for display
    error: str


# ────────────────────────────────────────────────────────────────
# Node 1 — input_parser
# ────────────────────────────────────────────────────────────────

def input_parser(state: AgentState) -> AgentState:
    """Validate and normalise the borrower_profile dict."""
    trace = list(state.get("agent_trace", []))
    profile = dict(state.get("borrower_profile", {}))

    errors = []

    # Required fields
    required_fields = [
        "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        "CNT_CHILDREN", "AMT_INCOME_TOTAL",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "JOB", "AGE", "EMPLOYED_YEARS",
    ]

    for f in required_fields:
        if f not in profile:
            errors.append(f"Missing required field: {f}")

    if errors:
        state["error"] = "; ".join(errors)
        trace.append({
            "node": "input_parser",
            "status": "❌ FAILED",
            "detail": state["error"],
        })
        state["agent_trace"] = trace
        return state

    # Normalise numeric fields
    try:
        profile["AMT_INCOME_TOTAL"] = float(profile["AMT_INCOME_TOTAL"])
        profile["AGE"] = float(profile["AGE"])
        profile["EMPLOYED_YEARS"] = float(profile["EMPLOYED_YEARS"])
    except (ValueError, TypeError) as e:
        state["error"] = f"Numeric conversion error: {e}"
        trace.append({
            "node": "input_parser",
            "status": "❌ FAILED",
            "detail": state["error"],
        })
        state["agent_trace"] = trace
        return state

    # Range checks
    if not (18 <= profile["AGE"] <= 80):
        errors.append(f"Age {profile['AGE']} out of valid range (18–80)")
    if profile["AMT_INCOME_TOTAL"] <= 0:
        errors.append("Income must be positive")
    if profile["EMPLOYED_YEARS"] < 0:
        errors.append("Employed years cannot be negative")

    if errors:
        state["error"] = "; ".join(errors)
        trace.append({
            "node": "input_parser",
            "status": "⚠️ VALIDATION WARNINGS",
            "detail": state["error"],
        })
        # Don't stop — proceed with warnings
        state["error"] = ""

    trace.append({
        "node": "input_parser",
        "status": "✅ OK",
        "detail": f"Validated {len(profile)} fields. Profile ready.",
    })

    state["borrower_profile"] = profile
    state["agent_trace"] = trace
    return state


# ────────────────────────────────────────────────────────────────
# Node 2 — risk_analyzer
# ────────────────────────────────────────────────────────────────

def risk_analyzer(state: AgentState) -> AgentState:
    """Run ML prediction & extract feature importances."""
    trace = list(state.get("agent_trace", []))

    # Check for prior errors that should halt
    if state.get("error"):
        trace.append({
            "node": "risk_analyzer",
            "status": "⏭️ SKIPPED",
            "detail": f"Upstream error: {state['error']}",
        })
        state["agent_trace"] = trace
        return state

    # Check that models are trained
    if "trained_models" not in st.session_state:
        state["error"] = "Models not trained. Please train models on the Model Training page first."
        trace.append({
            "node": "risk_analyzer",
            "status": "❌ FAILED",
            "detail": state["error"],
        })
        state["agent_trace"] = trace
        return state

    try:
        results = st.session_state["trained_models"]
        profile = state["borrower_profile"]

        # Use Decision Tree model (it doesn't need scaling)
        dt_info = results["Decision Tree"]
        model = dt_info["model"]
        feature_names = results["feature_names"]

        # Build input DataFrame matching the training features
        input_df = pd.DataFrame([profile])
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Align with training columns
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]

        # Predict
        probability = model.predict_proba(input_encoded)[0]
        risk_score = float(probability[1])  # probability of class 1 (rejection)

        # Risk class
        if risk_score < 0.3:
            risk_class = "Low Risk"
        elif risk_score <= 0.6:
            risk_class = "Medium Risk"
        else:
            risk_class = "High Risk"

        # Top feature importances
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:3]
        risk_drivers = [feature_names[i] for i in top_indices]

        state["risk_score"] = risk_score
        state["risk_class"] = risk_class
        state["risk_drivers"] = risk_drivers

        trace.append({
            "node": "risk_analyzer",
            "status": "✅ OK",
            "detail": (
                f"Risk Score: {risk_score:.4f} | Class: {risk_class} | "
                f"Top drivers: {', '.join(risk_drivers)}"
            ),
        })

    except Exception as e:
        state["error"] = f"Risk analysis failed: {e}\n{traceback.format_exc()}"
        trace.append({
            "node": "risk_analyzer",
            "status": "❌ FAILED",
            "detail": state["error"],
        })

    state["agent_trace"] = trace
    return state


# ────────────────────────────────────────────────────────────────
# Node 3 — regulation_retriever
# ────────────────────────────────────────────────────────────────

def regulation_retriever(state: AgentState) -> AgentState:
    """Retrieve relevant regulations using RAG."""
    trace = list(state.get("agent_trace", []))

    if state.get("error"):
        trace.append({
            "node": "regulation_retriever",
            "status": "⏭️ SKIPPED",
            "detail": f"Upstream error: {state['error']}",
        })
        state["agent_trace"] = trace
        return state

    try:
        # Initialize RAG (idempotent)
        initialize_rag()

        risk_class = state.get("risk_class", "Medium Risk")
        drivers = state.get("risk_drivers", [])
        top_driver = drivers[0] if drivers else "income"

        query = (
            f"Lending regulation for {risk_class} borrower. "
            f"Key risk factor: {top_driver}. "
            f"Credit card approval criteria and regulatory compliance."
        )

        regulations = retrieve_regulations(query, n_results=3)
        state["retrieved_regulations"] = regulations

        trace.append({
            "node": "regulation_retriever",
            "status": "✅ OK",
            "detail": f"Retrieved {len(regulations)} relevant regulation(s).",
        })

    except Exception as e:
        state["retrieved_regulations"] = []
        trace.append({
            "node": "regulation_retriever",
            "status": "⚠️ WARNING",
            "detail": f"Regulation retrieval failed: {e}. Continuing without regulations.",
        })

    state["agent_trace"] = trace
    return state


# ────────────────────────────────────────────────────────────────
# Node 4 — report_generator
# ────────────────────────────────────────────────────────────────

def report_generator(state: AgentState) -> AgentState:
    """Call the LLM (or rule-based fallback) to generate the report."""
    trace = list(state.get("agent_trace", []))

    if state.get("error"):
        trace.append({
            "node": "report_generator",
            "status": "⏭️ SKIPPED",
            "detail": f"Upstream error: {state['error']}",
        })
        state["agent_trace"] = trace
        return state

    try:
        report = generate_credit_report(
            borrower_profile=state["borrower_profile"],
            risk_score=state.get("risk_score", 0.5),
            risk_class=state.get("risk_class", "Medium Risk"),
            risk_drivers=state.get("risk_drivers", []),
            regulations=state.get("retrieved_regulations", []),
        )

        state["llm_report"] = report

        trace.append({
            "node": "report_generator",
            "status": "✅ OK",
            "detail": f"Report generated. Recommendation: {report.get('lending_recommendation', 'N/A')}",
        })

    except Exception as e:
        state["error"] = f"Report generation failed: {e}"
        state["llm_report"] = {}
        trace.append({
            "node": "report_generator",
            "status": "❌ FAILED",
            "detail": state["error"],
        })

    state["agent_trace"] = trace
    return state


# ────────────────────────────────────────────────────────────────
# Node 5 — output_formatter
# ────────────────────────────────────────────────────────────────

def output_formatter(state: AgentState) -> AgentState:
    """Clean and finalise the report dict."""
    trace = list(state.get("agent_trace", []))

    if state.get("error"):
        trace.append({
            "node": "output_formatter",
            "status": "⏭️ SKIPPED",
            "detail": f"Upstream error: {state['error']}",
        })
        state["agent_trace"] = trace
        return state

    report = dict(state.get("llm_report", {}))

    # Ensure all required keys exist
    required_keys = [
        "borrower_summary", "risk_analysis", "lending_recommendation",
        "recommended_action", "regulatory_references", "disclaimer",
    ]
    for key in required_keys:
        if key not in report:
            report[key] = "N/A"

    # Normalise recommendation
    rec = str(report.get("lending_recommendation", "")).strip()
    if "reject" in rec.lower():
        report["lending_recommendation"] = "Reject"
    elif "conditional" in rec.lower():
        report["lending_recommendation"] = "Conditional Approval"
    elif "approve" in rec.lower():
        report["lending_recommendation"] = "Approve"
    # else keep as-is

    # Add metadata
    report["risk_score"] = state.get("risk_score", 0.0)
    report["risk_class"] = state.get("risk_class", "Unknown")
    report["risk_drivers"] = state.get("risk_drivers", [])
    report["borrower_profile"] = state.get("borrower_profile", {})

    state["llm_report"] = report

    trace.append({
        "node": "output_formatter",
        "status": "✅ OK",
        "detail": "Report finalised and validated.",
    })

    state["agent_trace"] = trace
    return state


# ────────────────────────────────────────────────────────────────
# Routing helper
# ────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route to END if there is a critical error."""
    if state.get("error"):
        return "end"
    return "continue"


# ────────────────────────────────────────────────────────────────
# Build the LangGraph
# ────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("input_parser", input_parser)
    graph.add_node("risk_analyzer", risk_analyzer)
    graph.add_node("regulation_retriever", regulation_retriever)
    graph.add_node("report_generator", report_generator)
    graph.add_node("output_formatter", output_formatter)

    # Set entry point
    graph.set_entry_point("input_parser")

    # Conditional edges: stop early on error
    graph.add_conditional_edges(
        "input_parser",
        should_continue,
        {"continue": "risk_analyzer", "end": END},
    )
    graph.add_conditional_edges(
        "risk_analyzer",
        should_continue,
        {"continue": "regulation_retriever", "end": END},
    )
    graph.add_conditional_edges(
        "regulation_retriever",
        should_continue,
        {"continue": "report_generator", "end": END},
    )
    graph.add_conditional_edges(
        "report_generator",
        should_continue,
        {"continue": "output_formatter", "end": END},
    )
    graph.add_edge("output_formatter", END)

    return graph


# Compile once
_compiled_graph = _build_graph().compile()


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────

def run_agent(borrower_profile: dict) -> AgentState:
    """
    Execute the full agentic workflow for a given borrower profile.

    Parameters
    ----------
    borrower_profile : dict
        Must contain: CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY,
        CNT_CHILDREN, AMT_INCOME_TOTAL, NAME_EDUCATION_TYPE,
        NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, JOB, AGE, EMPLOYED_YEARS.

    Returns
    -------
    AgentState  with all fields populated after the workflow run.
    """
    initial_state: AgentState = {
        "borrower_profile": borrower_profile,
        "risk_score": 0.0,
        "risk_class": "",
        "risk_drivers": [],
        "retrieved_regulations": [],
        "llm_report": {},
        "agent_trace": [],
        "error": "",
    }

    # Run the compiled graph
    final_state = _compiled_graph.invoke(initial_state)
    return final_state


# ────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("This module requires Streamlit session state for model access.")
    print("Run it via the Streamlit app (Page 4 — AI Lending Assistant).")
