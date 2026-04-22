import os
from html import escape
import re
from datetime import datetime

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --bg-main: #1b2838;
        --bg-panel: #223043;
        --bg-panel-soft: #14202e;
        --bg-input: #f8fafc;
        --text-main: #e8edf3;
        --text-muted: #b8c4d6;
        --text-dark: #1f2937;
        --accent: #3b82f6;
        --accent-strong: #2563eb;
        --border-soft: #2d4057;
        --success: #00b09b;
        --danger: #eb3349;
    }

    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
            sans-serif;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-main);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    #MainMenu, footer {
        display: none !important;
    }

    [data-testid="stHeader"] {
        background: rgba(27, 40, 56, 0.9);
    }

    section[data-testid="stSidebar"] {
        background-color: #0e1621;
        border-right: 1px solid #263347;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: var(--bg-panel-soft);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 14px 18px !important;
        margin-bottom: 10px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .stButton > button,
    div[data-testid="stDownloadButton"] > button {
        background-color: var(--accent-strong);
        color: white !important;
        border: 1px solid var(--accent);
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
        min-height: 48px;
    }

    .stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #1d4ed8;
        border-color: var(--accent-strong);
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 40px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #3b82f6;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 24px;
        margin-bottom: 20px;
        text-align: center;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stTextArea"] label,
    .stMarkdown,
    .stCaption,
    p,
    li {
        color: var(--text-main) !important;
    }

    div[data-testid="stTextInput"] label p,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stSlider"] label p,
    div[data-testid="stTextArea"] label p {
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div,
    textarea {
        background: var(--bg-input) !important;
        color: var(--text-dark) !important;
        border: 1px solid #d6dfeb !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg,
    div[data-baseweb="select"] * {
        color: var(--text-dark) !important;
        fill: var(--text-dark) !important;
    }

    div[data-baseweb="popover"] * {
        color: var(--text-dark) !important;
    }

    div[data-baseweb="popover"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #d6dfeb !important;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.18) !important;
    }

    div[data-testid="stNumberInput"] button {
        color: var(--text-dark) !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] * {
        color: var(--accent) !important;
    }

    div[data-testid="stTabs"] {
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 4px 4px 0 4px;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 54px;
        padding: 0 24px;
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px 8px 0 0;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(37, 99, 235, 0.25) !important;
        color: #ffffff !important;
        border-bottom: 3px solid #3b82f6 !important;
    }

    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] details summary {
        color: var(--text-main) !important;
        font-weight: 700 !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.03) !important;
    }

    .result-approved {
        background: linear-gradient(135deg, rgba(0, 176, 155, 0.2), rgba(0, 176, 155, 0.05));
        border: 2px solid #00b09b;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #00b09b;
        box-shadow: 0 0 30px rgba(0, 176, 155, 0.3);
        margin: 20px 0;
    }

    .result-rejected {
        background: linear-gradient(135deg, rgba(235, 51, 73, 0.2), rgba(235, 51, 73, 0.05));
        border: 2px solid #eb3349;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #eb3349;
        box-shadow: 0 0 30px rgba(235, 51, 73, 0.3);
        margin: 20px 0;
    }

    .report-preview {
        background: #f8fafc;
        border: 1px solid #d9e2ef;
        border-radius: 18px;
        padding: 28px;
        margin-top: 10px;
        color: #1f2937 !important;
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.18);
    }

    .report-preview *,
    .report-preview p,
    .report-preview div,
    .report-preview li,
    .report-preview span,
    .report-preview h3,
    .report-preview h4 {
        color: #1f2937 !important;
    }

    .report-preview-header {
        text-align: center;
        border-bottom: 2px solid #dbe7f5;
        padding-bottom: 14px;
        margin-bottom: 18px;
    }

    .report-preview-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0 0 4px 0;
    }

    .report-preview-meta {
        font-size: 0.92rem;
        color: #526072 !important;
        margin: 0;
    }

    .report-preview-status {
        margin: 18px 0;
        padding: 16px 18px;
        border-radius: 14px;
        font-weight: 800;
        text-align: center;
        font-size: 1.05rem;
    }

    .report-preview-status.approve {
        background: rgba(0, 176, 155, 0.12);
        border: 1px solid rgba(0, 176, 155, 0.35);
        color: #047857 !important;
    }

    .report-preview-status.reject {
        background: rgba(235, 51, 73, 0.10);
        border: 1px solid rgba(235, 51, 73, 0.35);
        color: #b42318 !important;
    }

    .report-preview-status.conditional {
        background: rgba(245, 158, 11, 0.13);
        border: 1px solid rgba(245, 158, 11, 0.40);
        color: #b45309 !important;
    }

    .report-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 18px;
    }

    .report-chip {
        background: #ffffff;
        border: 1px solid #d9e2ef;
        border-radius: 12px;
        padding: 12px 14px;
    }

    .report-chip-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b !important;
        margin-bottom: 6px;
    }

    .report-chip-value {
        font-size: 1rem;
        font-weight: 700;
    }

    .report-preview h4 {
        margin: 0 0 12px 0;
        font-size: 1.05rem;
    }

    .report-preview p {
        margin: 0 0 10px 0;
        line-height: 1.7;
    }

    .report-note {
        color: #526072 !important;
        font-size: 0.92rem;
    }

    .report-section {
        background: #ffffff;
        border: 1px solid #d9e2ef;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .report-section ul {
        margin: 0;
        padding-left: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)



# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────

def remove_outliers_iqr(df, col):
    """Remove outliers using IQR method (same as notebook)."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]


@st.cache_data
def load_and_preprocess_data(filepath):
    """Load and preprocess data exactly as the notebook does."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

    # Replace "?" with NaN
    df.replace("?", np.nan, inplace=True)

    # Flip days columns to positive
    for col in ['DAYS_BIRTH', 'DAYS_EMPLOYED']:
        df[col] = -df[col]

    # Create derived features
    df["AGE"] = df['DAYS_BIRTH'] / 365
    df["EMPLOYED_YEARS"] = df['DAYS_EMPLOYED'] / 365

    # Drop unnecessary columns
    drop_cols = ["FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "ID"]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop DAYS columns (replaced by AGE and EMPLOYED_YEARS)
    for c in ["DAYS_BIRTH", "DAYS_EMPLOYED"]:
        if c in df_clean.columns:
            df_clean.drop(columns=[c], inplace=True)

    # Remove outliers using IQR on numerical columns
    num_cols = ["AMT_INCOME_TOTAL", "AGE", "EMPLOYED_YEARS"]
    for col in num_cols:
        if col in df_clean.columns:
            df_clean = remove_outliers_iqr(df_clean, col)

    # Drop leakage columns
    leakage_cols = ["BEGIN_MONTHS", "STATUS"]
    df_clean = df_clean.drop(columns=[c for c in leakage_cols if c in df_clean.columns])

    return df, df_clean


@st.cache_resource
def train_models(df_clean):
    """Train both models and return them with metrics."""
    X = df_clean.drop("TARGET", axis=1)
    y = df_clean["TARGET"]

    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model 1: Logistic Regression with SMOTE ---
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        scaler_smote = StandardScaler()
        X_train_res_scaled = scaler_smote.fit_transform(X_train_res)
        X_test_scaled_smote = scaler_smote.transform(X_test)

        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train_res_scaled, y_train_res)

        y_pred_log = log_model.predict(X_test_scaled_smote)
        y_prob_log = log_model.predict_proba(X_test_scaled_smote)[:, 1]

        log_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_log),
            "roc_auc": roc_auc_score(y_test, y_prob_log),
            "confusion_matrix": confusion_matrix(y_test, y_pred_log),
            "classification_report": classification_report(y_test, y_pred_log, output_dict=True),
            "y_test": y_test,
            "y_pred": y_pred_log,
            "y_prob": y_prob_log,
        }
        log_result = {"model": log_model, "scaler": scaler_smote, "metrics": log_metrics}
    except ImportError:
        # Fallback without SMOTE
        log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
        log_model.fit(X_train_scaled, y_train)
        y_pred_log = log_model.predict(X_test_scaled)
        y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]
        log_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_log),
            "roc_auc": roc_auc_score(y_test, y_prob_log),
            "confusion_matrix": confusion_matrix(y_test, y_pred_log),
            "classification_report": classification_report(y_test, y_pred_log, output_dict=True),
            "y_test": y_test,
            "y_pred": y_pred_log,
            "y_prob": y_prob_log,
        }
        log_result = {"model": log_model, "scaler": scaler, "metrics": log_metrics}

    # --- Model 2: Decision Tree ---
    dt_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    y_pred_dt = dt_model.predict(X_test)
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

    dt_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_dt),
        "roc_auc": roc_auc_score(y_test, y_prob_dt),
        "confusion_matrix": confusion_matrix(y_test, y_pred_dt),
        "classification_report": classification_report(y_test, y_pred_dt, output_dict=True),
        "y_test": y_test,
        "y_pred": y_pred_dt,
        "y_prob": y_prob_dt,
    }
    dt_result = {"model": dt_model, "scaler": None, "metrics": dt_metrics}

    return {
        "Logistic Regression (SMOTE)": log_result,
        "Decision Tree": dt_result,
        "feature_names": feature_names,
        "scaler": scaler,
    }


def render_metric_card(label, value, emoji="📊"):
    """Render a glassmorphic metric card."""
    if isinstance(value, float):
        display = f"{value:.4f}"
    else:
        display = str(value)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{emoji} {label}</div>
        <div class="metric-value">{display}</div>
    </div>
    """, unsafe_allow_html=True)


def ensure_models_trained():
    """Train models once per session so prediction and AI pages do not fail silently."""
    if "trained_models" not in st.session_state:
        with st.spinner("Training models for this session..."):
            st.session_state["trained_models"] = train_models(df_clean)
    return st.session_state["trained_models"]


def escape_html_text(value):
    return escape(str(value)).replace("\n", "<br>")


def build_pdf_style_preview(profile, risk_score, risk_class, risk_drivers, regs, report):
    recommendation = str(report.get("lending_recommendation", "Conditional Approval"))
    recommendation_lower = recommendation.lower()
    if "approve" in recommendation_lower and "conditional" not in recommendation_lower:
        status_class = "approve"
        status_text = f"STATUS: {escape_html_text(profile.get('NAME', 'Applicant').upper())}, PROPOSAL ACCEPTED"
    elif "reject" in recommendation_lower:
        status_class = "reject"
        status_text = f"STATUS: {escape_html_text(profile.get('NAME', 'Applicant').upper())}, PROPOSAL REJECTED"
    else:
        status_class = "conditional"
        status_text = f"STATUS: {escape_html_text(profile.get('NAME', 'Applicant').upper())}, CONDITIONAL APPROVAL"

    regulation_items = "".join(
        f"<li>{escape_html_text(reg)}</li>" for reg in (regs or report.get("regulatory_references", []))[:3]
    )
    if not regulation_items:
        regulation_items = "<li>Relevant RBI-style compliance notes were not available for this run.</li>"

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: transparent;
          color: #1f2937;
        }}
        .report-preview {{
          background: #f8fafc;
          border: 1px solid #d9e2ef;
          border-radius: 18px;
          padding: 28px;
          color: #1f2937;
          box-shadow: 0 18px 44px rgba(15, 23, 42, 0.18);
        }}
        .report-preview-header {{
          text-align: center;
          border-bottom: 2px solid #dbe7f5;
          padding-bottom: 14px;
          margin-bottom: 18px;
        }}
        .report-preview-title {{
          font-size: 1.55rem;
          font-weight: 800;
          margin: 0 0 4px 0;
        }}
        .report-preview-meta {{
          font-size: 0.92rem;
          color: #526072;
          margin: 0;
        }}
        .report-preview-status {{
          margin: 18px 0;
          padding: 16px 18px;
          border-radius: 14px;
          font-weight: 800;
          text-align: center;
          font-size: 1.05rem;
        }}
        .report-preview-status.approve {{
          background: rgba(0, 176, 155, 0.12);
          border: 1px solid rgba(0, 176, 155, 0.35);
          color: #047857;
        }}
        .report-preview-status.reject {{
          background: rgba(235, 51, 73, 0.10);
          border: 1px solid rgba(235, 51, 73, 0.35);
          color: #b42318;
        }}
        .report-preview-status.conditional {{
          background: rgba(245, 158, 11, 0.13);
          border: 1px solid rgba(245, 158, 11, 0.40);
          color: #b45309;
        }}
        .report-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
          margin-bottom: 18px;
        }}
        .report-chip {{
          background: #ffffff;
          border: 1px solid #d9e2ef;
          border-radius: 12px;
          padding: 12px 14px;
        }}
        .report-chip-label {{
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #64748b;
          margin-bottom: 6px;
        }}
        .report-chip-value {{
          font-size: 1rem;
          font-weight: 700;
        }}
        .report-section {{
          background: #ffffff;
          border: 1px solid #d9e2ef;
          border-radius: 14px;
          padding: 18px;
          margin-bottom: 14px;
        }}
        .report-section h4 {{
          margin: 0 0 12px 0;
          font-size: 1.05rem;
        }}
        .report-section p {{
          margin: 0 0 10px 0;
          line-height: 1.7;
        }}
        .report-section ul {{
          margin: 0;
          padding-left: 1.2rem;
        }}
        .report-note {{
          color: #526072;
          font-size: 0.92rem;
        }}
      </style>
    </head>
    <body>
      <div class="report-preview">
        <div class="report-preview-header">
          <div class="report-preview-title">AI Lending Assessment Report</div>
          <p class="report-preview-meta">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="report-preview-status {status_class}">
          {status_text}
        </div>

        <div class="report-grid">
          <div class="report-chip">
            <div class="report-chip-label">Risk Score</div>
            <div class="report-chip-value">{risk_score:.4f}</div>
          </div>
          <div class="report-chip">
            <div class="report-chip-label">Risk Class</div>
            <div class="report-chip-value">{escape_html_text(risk_class)}</div>
          </div>
          <div class="report-chip">
            <div class="report-chip-label">Key Drivers</div>
            <div class="report-chip-value">{escape_html_text(", ".join(risk_drivers) if risk_drivers else "N/A")}</div>
          </div>
          <div class="report-chip">
            <div class="report-chip-label">Recommendation</div>
            <div class="report-chip-value">{escape_html_text(recommendation)}</div>
          </div>
        </div>

        <div class="report-section">
          <h4>1. Borrower Profile</h4>
          <p><strong>Name:</strong> {escape_html_text(profile.get("NAME", "Applicant"))}</p>
          <p><strong>Occupation:</strong> {escape_html_text(profile.get("JOB", "N/A"))}</p>
          <p><strong>Annual Income:</strong> INR {float(profile.get("AMT_INCOME_TOTAL", 0)):,.0f}</p>
          <p><strong>Education:</strong> {escape_html_text(profile.get("NAME_EDUCATION_TYPE", "N/A"))}</p>
          <p><strong>Housing:</strong> {escape_html_text(profile.get("NAME_HOUSING_TYPE", "N/A"))}</p>
        </div>

        <div class="report-section">
          <h4>2. Borrower Summary</h4>
          <p>{escape_html_text(report.get("borrower_summary", "N/A"))}</p>
        </div>

        <div class="report-section">
          <h4>3. Risk Analysis</h4>
          <p>{escape_html_text(report.get("risk_analysis", "N/A"))}</p>
        </div>

        <div class="report-section">
          <h4>4. Recommended Action</h4>
          <p>{escape_html_text(report.get("recommended_action", "N/A"))}</p>
        </div>

        <div class="report-section">
          <h4>5. Regulatory Context</h4>
          <ul>{regulation_items}</ul>
        </div>

        <div class="report-section">
          <h4>6. Disclaimer</h4>
          <p class="report-note">{escape_html_text(report.get("disclaimer", "N/A"))}</p>
        </div>
      </div>
    </body>
    </html>
    """


def extract_number(text, keywords, default):
    for keyword in keywords:
        pattern = rf"(?:{keyword})(?:\s+\w+){{0,4}}\s*(?:of|is|if|=|:|-)?\s*([\d][\d,.]*)"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1).replace(",", "").strip()
            if value:
                return float(value)
    return default


def extract_total_income(text, default):
    patterns = [
        r"total income\s*(?:of|is|=|:|-)?\s*([\d][\d,.]*)",
        r"income\s*(?:of|is|=|:|-)?\s*([\d][\d,.]*)",
        r"salary\s*(?:of|is|=|:|-)?\s*([\d][\d,.]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    return default


def extract_age_value(text, default):
    patterns = [
        r"age\s*(?:of|is|if|=|:|-)?\s*(\d{1,2}(?:\.\d+)?)",
        r"(\d{1,2})\s*[- ]?year[s]?\s*old",
        r"aged\s*(\d{1,2}(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return default


def parse_children(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["no child", "0 child", "without children", "no children"]):
        return "No children"
    if any(token in lowered for token in ["has a son", "has a daughter", "one son", "one daughter", "a child"]):
        return "1 children"
    if re.search(r"\b1\s+child\b", lowered):
        return "1 children"
    if re.search(r"\b([2-9]|\d{2,})\s+children?\b", lowered):
        return "2+ children"
    return "No children"


def parse_borrower_query(query_text: str) -> dict:
    q = query_text.lower()

    name_match = re.search(
        r"(?:name\s*(?:is|:)?|myself|i\s*am|this\s*is)\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)",
        query_text,
        re.IGNORECASE,
    )
    extracted_name = name_match.group(1).strip().title() if name_match else "Applicant"
    if extracted_name == "Applicant":
        first_word_match = re.match(r"\s*([A-Za-z]+)", query_text)
        if first_word_match:
            extracted_name = first_word_match.group(1).strip().title()

    gender = "F" if any(word in q for word in ["female", "woman", "girl"]) else "M"
    if any(word in q for word in ["male", "man", "boy"]):
        gender = "M"

    own_car = "N" if any(phrase in q for phrase in ["no car", "does not own a car", "without car"]) else "Y"
    if not any(word in q for word in ["car", "vehicle"]):
        own_car = "N"

    own_realty = "N" if any(
        phrase in q for phrase in ["no house", "no property", "rented apartment", "with parents", "without house"]
    ) else "Y"
    if not any(word in q for word in ["house", "apartment", "property", "realty", "parents", "rent"]):
        own_realty = "N"

    income = extract_total_income(q, 200000)
    age = extract_age_value(q, 35)
    employed = extract_number(q, ["employed", "experience", "working", "years employed"], 5)

    education_map = {
        "academic degree": "Academic degree",
        "higher education": "Higher education",
        "graduate": "Higher education",
        "incomplete higher": "Incomplete higher",
        "lower secondary": "Lower secondary",
        "secondary": "Secondary / secondary special",
    }
    education_q = "Secondary / secondary special"
    for key, value in education_map.items():
        if key in q:
            education_q = value
            break

    family_map = {
        "married": "Married",
        "single": "Single / not married",
        "civil marriage": "Civil marriage",
        "separated": "Separated",
        "widow": "Widow",
        "widowed": "Widow",
    }
    family_q = "Single / not married"
    for key, value in family_map.items():
        if key in q:
            family_q = value
            break

    housing_map = {
        "with parents": "With parents",
        "rented apartment": "Rented apartment",
        "rent": "Rented apartment",
        "municipal": "Municipal apartment",
        "co-op": "Co-op apartment",
        "office apartment": "Office apartment",
        "house": "House / apartment",
        "apartment": "House / apartment",
        "property": "House / apartment",
    }
    housing_q = "House / apartment"
    for key, value in housing_map.items():
        if key in q:
            housing_q = value
            break

    job_list = [
        "Managers", "Private service staff", "Laborers", "Core staff",
        "Drivers", "High skill tech staff", "Realty agents", "Secretaries",
        "Accountants", "Sales staff", "Medicine staff", "Waiters/barmen staff",
        "Low-skill Laborers", "Cleaning staff", "HR staff", "Cooking staff",
        "Security staff", "IT staff",
    ]
    job_q = "Laborers"
    for job_name in job_list:
        if job_name.lower() in q:
            job_q = job_name
            break

    return {
        "NAME": extracted_name,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": parse_children(q),
        "AMT_INCOME_TOTAL": float(income),
        "NAME_EDUCATION_TYPE": education_q,
        "NAME_FAMILY_STATUS": family_q,
        "NAME_HOUSING_TYPE": housing_q,
        "JOB": job_q,
        "AGE": float(age),
        "EMPLOYED_YEARS": float(employed),
    }


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💳 Credit Card Approval Predictor")
    st.markdown("<hr style='border-color:#263347; margin:12px 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Data Exploration", "🤖 Model Training", "🔮 Predict Approval", "🤖 AI Lending Assistant"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#263347; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#4a6178; font-size:0.75rem;'>"
        "ML Project &middot; Streamlit</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "credit_card_approval.csv")

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found! Please place `credit_card_approval.csv` in the same folder as this app.")
    st.stop()

df_raw, df_clean = load_and_preprocess_data(DATA_PATH)


# ══════════════════════════════════════════════
# PAGE 1: Data Exploration
# ══════════════════════════════════════════════
if page == "📊 Data Exploration":
    st.markdown("# 📊 Data Exploration")
    st.markdown("Explore the Credit Card Approval dataset — distributions, correlations, and insights.")

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Records", f"{len(df_raw):,}", "📁")
    with col2:
        render_metric_card("After Cleaning", f"{len(df_clean):,}", "🧹")
    with col3:
        render_metric_card("Features", df_clean.shape[1] - 1, "🔢")
    with col4:
        approval_rate = (df_clean["TARGET"].value_counts().get(0, 0) / len(df_clean)) * 100
        render_metric_card("Approval Rate", f"{approval_rate:.1f}%", "✅")

    st.markdown("")

    # --- Dataset Preview ---
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔍 Sample Data", "📐 Statistics"])

    with tab1:
        st.dataframe(df_clean.head(20), width="stretch", height=400)

    with tab2:
        st.dataframe(df_clean.describe(), width="stretch")

    # --- Target Distribution ---
    st.markdown('<div class="section-header">🎯 Target Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        target_counts = df_clean["TARGET"].value_counts().reset_index()
        target_counts.columns = ["Target", "Count"]
        target_counts["Target"] = target_counts["Target"].map({0: "Approved (0)", 1: "Rejected (1)"})
        fig = px.pie(
            target_counts, values="Count", names="Target",
            color_discrete_sequence=["#00b09b", "#eb3349"],
            hole=0.5,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(
            target_counts, x="Target", y="Count",
            color="Target",
            color_discrete_sequence=["#00b09b", "#eb3349"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
            xaxis=dict(showgrid=False, tickfont=dict(color="white")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="white")),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, width="stretch")

    # --- Feature Distributions ---
    st.markdown('<div class="section-header">📈 Feature Distributions</div>', unsafe_allow_html=True)

    num_cols = df_clean.select_dtypes(include="number").columns.drop("TARGET", errors="ignore").tolist()
    cat_cols = df_clean.select_dtypes(exclude="number").columns.tolist()

    tab_num, tab_cat = st.tabs(["🔢 Numerical Features", "🏷️ Categorical Features"])

    with tab_num:
        selected_num = st.selectbox("Select a numerical feature", num_cols, key="num_feat")
        fig = px.histogram(
            df_clean, x=selected_num, color=df_clean["TARGET"].map({0: "Approved", 1: "Rejected"}),
            barmode="overlay", nbins=50,
            color_discrete_sequence=["#667eea", "#eb3349"],
            labels={"color": "Status"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, width="stretch")

    with tab_cat:
        if cat_cols:
            selected_cat = st.selectbox("Select a categorical feature", cat_cols, key="cat_feat")
            cat_data = df_clean[selected_cat].value_counts().reset_index()
            cat_data.columns = [selected_cat, "Count"]
            fig = px.bar(
                cat_data, x=selected_cat, y="Count",
                color=selected_cat,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(showgrid=False, tickfont=dict(color="white")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="white")),
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No categorical features available.")

    # --- Correlation Heatmap ---
    st.markdown('<div class="section-header">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df_clean[num_cols + ["TARGET"]].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════
# PAGE 2: Model Training & Evaluation
# ══════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("# 🤖 Model Training & Evaluation")
    st.markdown(
        "Train Logistic Regression (with SMOTE) and Decision Tree models, "
        "then compare their performance."
    )

    if st.button("🚀 Train Models", type="primary", width="stretch"):
        with st.spinner("Training models... This may take a moment ⏳"):
            results = train_models(df_clean)
            st.session_state["trained_models"] = results
        st.success("✅ Models trained successfully!")

    if "trained_models" in st.session_state:
        results = st.session_state["trained_models"]

        # --- Model Comparison ---
        st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

        model_names = [k for k in results if k not in ("feature_names", "scaler")]

        cols = st.columns(len(model_names))
        for i, name in enumerate(model_names):
            metrics = results[name]["metrics"]
            with cols[i]:
                st.markdown(f"#### {name}")
                render_metric_card("Accuracy", metrics["accuracy"], "🎯")
                render_metric_card("ROC-AUC", metrics["roc_auc"], "📈")

        # --- Confusion Matrices ---
        st.markdown('<div class="section-header">🔢 Confusion Matrices</div>', unsafe_allow_html=True)

        cols = st.columns(len(model_names))
        for i, name in enumerate(model_names):
            cm = results[name]["metrics"]["confusion_matrix"]
            with cols[i]:
                st.markdown(f"**{name}**")
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Approved", "Rejected"],
                    y=["Approved", "Rejected"],
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    xaxis=dict(tickfont=dict(color="white")),
                    yaxis=dict(tickfont=dict(color="white")),
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                st.plotly_chart(fig, width="stretch")

        # --- ROC Curves ---
        st.markdown('<div class="section-header">📉 ROC Curves</div>', unsafe_allow_html=True)

        fig = go.Figure()
        colors = ["#667eea", "#f45c43"]
        for i, name in enumerate(model_names):
            m = results[name]["metrics"]
            fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{name} (AUC={m["roc_auc"]:.4f})',
                line=dict(color=colors[i % len(colors)], width=3),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="white")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="white")),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="white")),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, width="stretch")

        # --- Classification Reports ---
        st.markdown('<div class="section-header">📋 Classification Reports</div>', unsafe_allow_html=True)

        for name in model_names:
            report = results[name]["metrics"]["classification_report"]
            st.markdown(f"**{name}**")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), width="stretch")

    else:
        st.info("👆 Click **Train Models** to begin.")


# ══════════════════════════════════════════════
# PAGE 3: Predict Approval
# ══════════════════════════════════════════════
elif page == "🔮 Predict Approval":
    st.markdown("# 🔮 Predict Credit Card Approval")
    st.markdown("Enter applicant details to predict if the credit card will be **approved** or **rejected**.")

    if "trained_models" not in st.session_state:
        st.info("Models were not trained yet for this session. Training them automatically now.")
        ensure_models_trained()

    results = st.session_state["trained_models"]
    model_names = [k for k in results if k not in ("feature_names", "scaler")]

    # --- Model Selector ---
    selected_model = st.selectbox("Select Model", model_names, key="pred_model")

    st.markdown('<div class="section-header">📝 Applicant Details</div>', unsafe_allow_html=True)

    # --- Input Form ---
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["F", "M"])
        own_car = st.selectbox("Owns a Car?", ["Y", "N"])
        own_realty = st.selectbox("Owns Realty?", ["Y", "N"])
        children = st.selectbox("Number of Children", ["No children", "1 children", "2+ children"])

    with col2:
        income = st.number_input("Annual Income (₹)", min_value=0, max_value=10000000, value=200000, step=10000)
        age = st.slider("Age (years)", min_value=18, max_value=70, value=35)
        employed_years = st.slider("Years Employed", min_value=0.0, max_value=40.0, value=5.0, step=0.5)

    with col3:
        education = st.selectbox("Education", [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ])
        family_status = st.selectbox("Family Status", [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow",
        ])
        housing = st.selectbox("Housing Type", [
            "House / apartment",
            "With parents",
            "Rented apartment",
            "Municipal apartment",
            "Co-op apartment",
            "Office apartment",
        ])

    job = st.selectbox("Job Type", [
        "Managers", "Private service staff", "Laborers", "Core staff",
        "Drivers", "High skill tech staff", "Realty agents", "Secretaries",
        "Accountants", "Sales staff", "Medicine staff", "Waiters/barmen staff",
        "Low-skill Laborers", "Cleaning staff", "HR staff", "Cooking staff",
        "Security staff", "IT staff",
    ])

    st.markdown("")

    if st.button("🔮 Predict", type="primary", width="stretch"):
        try:
            # Build input dict
            input_data = {
                "CODE_GENDER": gender,
                "FLAG_OWN_CAR": own_car,
                "FLAG_OWN_REALTY": own_realty,
                "CNT_CHILDREN": children,
                "AMT_INCOME_TOTAL": float(income),
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family_status,
                "NAME_HOUSING_TYPE": housing,
                "JOB": job,
                "AGE": float(age),
                "EMPLOYED_YEARS": float(employed_years),
            }

            input_df = pd.DataFrame([input_data])

            # One-hot encode to match training features
            input_encoded = pd.get_dummies(input_df, drop_first=True)

            # Align columns with training data
            feature_names = results["feature_names"]
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[feature_names]

            # Get model & scaler
            model_info = results[selected_model]
            model = model_info["model"]
            scaler = model_info["scaler"]

            # Scale if needed
            if scaler is not None:
                input_scaled = scaler.transform(input_encoded)
            else:
                input_scaled = input_encoded

            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.stop()

        st.markdown("")

        # --- Display Result ---
        col1, col2 = st.columns([2, 1])

        with col1:
            if prediction == 0:
                st.markdown(
                    '<div class="result-approved">'
                    '✅ APPROVED<br>'
                    '<span style="font-size:1rem;font-weight:400;">Credit card application is likely to be approved!</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-rejected">'
                    '❌ REJECTED<br>'
                    '<span style="font-size:1rem;font-weight:400;">Credit card application is likely to be rejected.</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        with col2:
            # Confidence gauge
            confidence = max(probability) * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Confidence", "font": {"color": "white"}},
                number={"suffix": "%", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#667eea"},
                    "bgcolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(235, 51, 73, 0.3)"},
                        {"range": [50, 75], "color": "rgba(255, 193, 7, 0.3)"},
                        {"range": [75, 100], "color": "rgba(0, 176, 155, 0.3)"},
                    ],
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=250,
                margin=dict(t=40, b=20, l=30, r=30),
            )
            st.plotly_chart(fig, width="stretch")

        # Probability breakdown
        st.markdown('<div class="section-header">📊 Probability Breakdown</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Class": ["Approved (0)", "Rejected (1)"],
            "Probability": probability,
        })
        fig = px.bar(
            prob_df, x="Class", y="Probability",
            color="Class",
            color_discrete_sequence=["#00b09b", "#eb3349"],
            text=prob_df["Probability"].apply(lambda x: f"{x:.4f}"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", range=[0, 1]),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════
# PAGE 4: AI Lending Assistant
# ══════════════════════════════════════════════
elif page == "🤖 AI Lending Assistant":
    st.markdown("# 🤖 AI Lending Assistant")
    st.markdown(
        "Agentic AI-powered credit assessment using ML models, regulatory RAG, "
        "and LLM-generated reports."
    )

    # ── helper: display the full report ──
    def display_agent_report(final_state):
        """Render the complete AI assessment report from agent state."""
        report = final_state.get("llm_report", {})
        profile = final_state.get("borrower_profile", {})
        risk_score = final_state.get("risk_score", 0.0)
        risk_class = final_state.get("risk_class", "Unknown")
        risk_drivers = final_state.get("risk_drivers", [])
        regs = final_state.get("retrieved_regulations", [])
        trace = final_state.get("agent_trace", [])
        error = final_state.get("error", "")

        if error:
            st.error(f"❌ Agent error: {error}")
            return

        # ── Agent Trace ──
        with st.expander("🔍 Agent Execution Trace", expanded=False):
            for step in trace:
                status_icon = step.get("status", "")
                node_name = step.get("node", "")
                detail = step.get("detail", "")
                st.markdown(
                    f"**{status_icon}  `{node_name}`** — {detail}"
                )

        st.markdown("")

        # ── Borrower Profile Card ──
        st.markdown('<div class="section-header">📋 Borrower Profile</div>', unsafe_allow_html=True)
        profile_cols = st.columns(3)
        items = list(profile.items())
        for i, (k, v) in enumerate(items):
            with profile_cols[i % 3]:
                render_metric_card(k.replace("_", " ").title(), v, "👤")

        st.markdown("")

        # ── Risk Score & Class ──
        st.markdown('<div class="section-header">🎯 Risk Score & Classification</div>', unsafe_allow_html=True)
        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            confidence = (1 - risk_score) * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={"text": "Default Probability (%)", "font": {"color": "white"}},
                number={"suffix": "%", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#667eea"},
                    "bgcolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(0, 176, 155, 0.3)"},
                        {"range": [30, 60], "color": "rgba(255, 193, 7, 0.3)"},
                        {"range": [60, 100], "color": "rgba(235, 51, 73, 0.3)"},
                    ],
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=280,
                margin=dict(t=40, b=20, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, width="stretch")
        with r_col2:
            rc_color = (
                "#00b09b" if risk_class == "Low Risk"
                else ("#f59e0b" if risk_class == "Medium Risk" else "#eb3349")
            )
            st.markdown(
                f'<div class="metric-card" style="border-color:{rc_color};margin-top:30px;">'
                f'<div class="metric-label">Risk Class</div>'
                f'<div class="metric-value" style="color:{rc_color};font-size:1.6rem;">{risk_class}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Top Risk Drivers ──
        if risk_drivers:
            st.markdown('<div class="section-header">⚠️ Top Risk Drivers</div>', unsafe_allow_html=True)
            try:
                results_m = st.session_state.get("trained_models", {})
                dt_info = results_m.get("Decision Tree", {})
                model_dt = dt_info.get("model")
                feat_names = results_m.get("feature_names", [])
                if model_dt is not None and hasattr(model_dt, "feature_importances_"):
                    importances = model_dt.feature_importances_
                    top_n = min(5, len(feat_names))
                    top_idx = np.argsort(importances)[::-1][:top_n]
                    top_features = [feat_names[i] for i in top_idx]
                    top_values = [float(importances[i]) for i in top_idx]
                    fig_bar = go.Figure(go.Bar(
                        x=top_values[::-1],
                        y=top_features[::-1],
                        orientation="h",
                        marker_color="#667eea",
                    ))
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                                   title="Feature Importance"),
                        yaxis=dict(showgrid=False),
                        height=250,
                        margin=dict(t=10, b=30, l=20, r=20),
                    )
                    st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info("Feature importance data not available.")
            except Exception:
                for d in risk_drivers:
                    st.markdown(f"- **{d}**")

        # ── Regulatory Context ──
        if regs:
            st.markdown('<div class="section-header">📜 Regulatory Context</div>', unsafe_allow_html=True)
            for reg in regs:
                st.markdown(
                    f'<div style="background:rgba(30,58,95,0.6);border-left:4px solid #3b82f6;'
                    f'padding:16px 20px;border-radius:8px;margin-bottom:12px;'
                    f'color:#cbd5e1;font-size:0.92rem;">{reg}</div>',
                    unsafe_allow_html=True,
                )

        # ── Lending Recommendation ──
        rec = report.get("lending_recommendation", "N/A")
        st.markdown('<div class="section-header">✅ Lending Recommendation</div>', unsafe_allow_html=True)
        name = profile.get("NAME", "Applicant")
        if rec == "Approve":
            st.markdown(
                '<div class="result-approved">'
                '✅ PROPOSAL ACCEPTED<br>'
                f'<span style="font-size:1rem;font-weight:700;">Dear {name}, your application has been ACCEPTED.</span><br>'
                '<span style="font-size:1rem;font-weight:400;">Application meets lending criteria.</span>'
                '</div>', unsafe_allow_html=True,
            )
        elif rec == "Reject":
            st.markdown(
                '<div class="result-rejected">'
                '❌ PROPOSAL REJECTED<br>'
                f'<span style="font-size:1rem;font-weight:700;">Dear {name}, your application has been REJECTED.</span><br>'
                '<span style="font-size:1rem;font-weight:400;">Application does not meet lending criteria.</span>'
                '</div>', unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(245,158,11,0.2),rgba(245,158,11,0.05));'
                f'border:2px solid #f59e0b;border-radius:20px;padding:40px;text-align:center;'
                f'font-size:2.5rem;font-weight:800;color:#f59e0b;'
                f'box-shadow:0 0 30px rgba(245,158,11,0.3);margin:20px 0;">'
                f'⚠️ PROPOSAL: CONDITIONAL APPROVAL<br>'
                f'<span style="font-size:1rem;font-weight:400;">Additional review required.</span>'
                f'</div>', unsafe_allow_html=True,
            )

        # ── AI Analysis ──
        st.markdown('<div class="section-header">💡 AI Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);'
            f'border-radius:12px;padding:24px;margin-bottom:12px;">'
            f'<p style="color:#e8edf3;font-size:0.95rem;line-height:1.7;">'
            f'<strong>Summary:</strong> {escape_html_text(report.get("borrower_summary", "N/A"))}</p>'
            f'<p style="color:#e8edf3;font-size:0.95rem;line-height:1.7;margin-top:12px;">'
            f'<strong>Risk Analysis:</strong> {escape_html_text(report.get("risk_analysis", "N/A"))}</p>'
            f'<p style="color:#cbd5e1;font-size:0.95rem;line-height:1.7;margin-top:12px;">'
            f'<strong>Recommended Action:</strong> {escape_html_text(report.get("recommended_action", "N/A"))}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Disclaimer ──
        st.markdown(
            f'<div style="background:rgba(100,100,100,0.15);border-radius:8px;'
            f'padding:16px 20px;margin-top:20px;color:#718096;font-size:0.82rem;">'
            f'⚖️ <strong>Disclaimer:</strong> {escape_html_text(report.get("disclaimer", "N/A"))}</div>',
            unsafe_allow_html=True,
        )

        # ── Download Report ──
        st.markdown("")
        from pdf_generator import create_pdf_report
        pdf_bytes = create_pdf_report(profile, risk_score, risk_class, risk_drivers, regs, report)
        
        st.download_button(
            label="📥 Download Full Report (PDF)",
            data=pdf_bytes,
            file_name="ai_lending_report.pdf",
            mime="application/pdf",
            on_click="ignore",
            use_container_width=True,
        )

        with st.expander("📄 Report Preview", expanded=True):
            components.html(
                build_pdf_style_preview(profile, risk_score, risk_class, risk_drivers, regs, report),
                height=1400,
                scrolling=True,
            )



    # ── Tabs ──
    tab1, tab2 = st.tabs(["💬 Query Mode", "📋 Form Assessment"])

    # ──────────────────────────────────────────────
    # Tab 1 — Query Mode
    # ──────────────────────────────────────────────
    with tab1:
        st.markdown("Enter a freeform description of a loan applicant. "
                    "The AI agent will parse the query, run ML prediction, "
                    "retrieve regulations, and generate a full report.")

        query_text = st.text_area(
            "Describe the borrower",
            placeholder=(
                "Example: 35-year-old married male, income 300000, higher education, "
                "works as a manager, owns a car and house, no children, employed 8 years."
            ),
            height=120,
        )

        if st.button("🚀 Run Agent", key="query_run", type="primary"):
            valid_kws = ["income", "salary", "age", "year", "employed", "car", "house", "loan"]
            q = query_text.lower()
            if "trained_models" not in st.session_state:
                st.info("Models were not trained yet for this session. Training them automatically now.")
                ensure_models_trained()
            if not query_text.strip() or not any(kw in q for kw in valid_kws):
                st.error("Invalid input. Please provide valid borrower details like age, income, and employment.")
            else:
                parsed_profile = parse_borrower_query(query_text)

                st.markdown("**Parsed profile:**")
                profile_html = "<div style='background:rgba(255,255,255,0.05); padding:15px; border-radius:10px; font-family:monospace; font-size:14px; color:#cbd5e1;'>"
                for k, v in parsed_profile.items():
                    profile_html += (
                        f"<span style='color:#3b82f6; font-weight:700;'>\"{escape_html_text(k)}\"</span>: "
                        f"{escape_html_text(v)}<br>"
                    )
                profile_html += "</div><br>"
                st.markdown(profile_html, unsafe_allow_html=True)

                try:
                    from agent_workflow import run_agent
                    with st.spinner("🤖 Agent is analyzing the borrower profile..."):
                        st.session_state["agent_state"] = run_agent(parsed_profile)
                except Exception as e:
                    st.error(f"❌ Agent execution failed: {e}")

    # ──────────────────────────────────────────────
    # Tab 2 — Form Assessment
    # ──────────────────────────────────────────────
    with tab2:
        if "trained_models" not in st.session_state:
            st.info("Models were not trained yet for this session. Training them automatically now.")
            ensure_models_trained()

        st.markdown('<div class="section-header">📝 Applicant Details</div>', unsafe_allow_html=True)
        applicant_name_f = st.text_input("Enter your name", key="form_name")

        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            f_gender = st.selectbox("Gender", ["F", "M"], key="f_gender")
            f_car = st.selectbox("Owns a Car?", ["Y", "N"], key="f_car")
            f_realty = st.selectbox("Owns Realty?", ["Y", "N"], key="f_realty")
            f_children = st.selectbox("Number of Children",
                                      ["No children", "1 children", "2+ children"],
                                      key="f_children")

        with f_col2:
            f_income = st.number_input("Annual Income (₹)", min_value=0,
                                        max_value=10000000, value=200000,
                                        step=10000, key="f_income")
            f_age = st.slider("Age (years)", min_value=18, max_value=70,
                               value=35, key="f_age")
            f_employed = st.slider("Years Employed", min_value=0.0,
                                    max_value=40.0, value=5.0, step=0.5,
                                    key="f_employed")

        with f_col3:
            f_education = st.selectbox("Education", [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ], key="f_education")
            f_family = st.selectbox("Family Status", [
                "Married",
                "Single / not married",
                "Civil marriage",
                "Separated",
                "Widow",
            ], key="f_family")
            f_housing = st.selectbox("Housing Type", [
                "House / apartment",
                "With parents",
                "Rented apartment",
                "Municipal apartment",
                "Co-op apartment",
                "Office apartment",
            ], key="f_housing")

        f_job = st.selectbox("Job Type", [
            "Managers", "Private service staff", "Laborers", "Core staff",
            "Drivers", "High skill tech staff", "Realty agents", "Secretaries",
            "Accountants", "Sales staff", "Medicine staff", "Waiters/barmen staff",
            "Low-skill Laborers", "Cleaning staff", "HR staff", "Cooking staff",
            "Security staff", "IT staff",
        ], key="f_job")

        st.markdown("")

        if st.button("🚀 Run AI Assessment", key="form_run", type="primary"):
            if not applicant_name_f.strip():
                st.error("Invalid input. Please provide all required and valid details.")
            else:
                form_profile = {
                    "NAME": applicant_name_f.strip(),
                    "CODE_GENDER": f_gender,
                    "FLAG_OWN_CAR": f_car,
                    "FLAG_OWN_REALTY": f_realty,
                    "CNT_CHILDREN": f_children,
                    "AMT_INCOME_TOTAL": float(f_income),
                    "NAME_EDUCATION_TYPE": f_education,
                    "NAME_FAMILY_STATUS": f_family,
                    "NAME_HOUSING_TYPE": f_housing,
                    "JOB": f_job,
                    "AGE": float(f_age),
                    "EMPLOYED_YEARS": float(f_employed),
                }

                try:
                    from agent_workflow import run_agent
                    with st.spinner("🤖 Agent is analyzing the borrower profile..."):
                        st.session_state["agent_state"] = run_agent(form_profile)
                except Exception as e:
                    st.error(f"❌ Agent execution failed: {e}")

    # ── Render active cross-tab state ──
    if st.session_state.get("agent_state"):
        st.markdown("<hr>", unsafe_allow_html=True)
        display_agent_report(st.session_state["agent_state"])
