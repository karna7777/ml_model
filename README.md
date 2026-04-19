# Intelligent Credit Risk Scoring & Agentic Lending Decision Support

## 📋 Project Overview
This project presents an intelligent credit risk scoring system coupled with an agentic lending decision support framework. It leverages predictive machine learning models to assess borrower risk and an LLM-powered multi-agent system to interpret the results within the context of Reserve Bank of India (RBI) regulations. By synthesizing quantitative risk scores with qualitative regulatory analysis, it aims to provide objective, actionable, and compliant lending recommendations. The project was developed across two key milestones: implementing the core ML risk pipeline and integrating the advanced Agentic workflow.

## 🛠 Tech Stack

| Component | Technology |
| --- | --- |
| **Language** | Python |
| **ML Models** | Logistic Regression (SMOTE), Decision Tree |
| **Agentic Framework** | LangGraph |
| **RAG/Vector Store** | ChromaDB, all-MiniLM-L6-v2 embeddings |
| **LLM Inference** | Groq API (llama-3.1-8b-instant) |
| **UI Framework** | Streamlit (4 pages) |
| **PDF Export**| fpdf2 |
| **Hosting** | Streamlit Cloud / Hugging Face Spaces |

## 📥 Input / Output

**Inputs:**
- Applicant financial profile (e.g., income, credit history, outstanding debt)
- Demographics & employment specifics (based on Kaggle Credit Card Approval dataset features)

**Outputs:**
- **Risk Score:** Quantitative probability of default.
- **Risk Class:** Categorical rating (e.g., Low, Medium, High Risk).
- **Structured Report:** Comprehensive PDF summary including analysis and regulatory compliance notes.

## 🏗 System Architecture

```text
[ Input Features ] 
        │
        ▼
[ ML Layer (LogReg / Decision Tree) ] ──▶ [ Risk Score & Drivers ]
        │                                         │
        ▼                                         ▼
[ Agentic Layer (LangGraph + RAG + LLM) ] ◀───────┘
        │
        ▼
[ Final Output: Formatted decision report & PDF ]
```

## 🧠 ML Pipeline (Milestone 1)

The machine learning pipeline processes raw applicant data to produce robust risk assessments, addressing inherent class imbalances in the dataset.

- **Preprocessing:** Data cleaning, handling missing values, standard scaling for numerical features, and one-hot encoding for categorical variables.
- **Models Used:** 
  - Logistic Regression (utilizing SMOTE to handle class imbalance).
  - Decision Tree classifier (`max_depth=5`, `class_weight='balanced'`).

**Performance Metrics**

| Model | Accuracy | ROC-AUC | F1-Score |
| --- | --- | --- | --- |
| Logistic Regression (SMOTE) | 0.xx | 0.xx | 0.xx |
| Decision Tree (Balanced) | 0.xx | 0.xx | 0.xx |
*(Note: Replace 0.xx with final calculated metrics)*

**Top 5 Risk Drivers (Decision Tree Feature Importance):**
1. [Feature 1, e.g., Debt-to-Income Ratio]
2. [Feature 2, e.g., Payment History Status]
3. [Feature 3, e.g., Total Income]
4. [Feature 4, e.g., Number of Open Credit Lines]
5. [Feature 5, e.g., Credit Utilization]
*(Note: Update with actual feature names)*

## 🤖 Agentic Workflow (Milestone 2)

The core decision support logic is orchestrated using LangGraph, executing a deterministic state machine that queries an LLM and local RAG database.

### Node Architecture
```text
(Start)
   │
   ▼
[1: input_parser] ───────▶ (Validates JSON/format)
   │
   ▼
[2: risk_analyzer] ──────▶ (Interprets ML score & drivers)
   │
   ▼
[3: regulation_retriever]▶ (ChromaDB RAG: Fetches top 3 RBI guidelines)
   │
   ▼
[4: report_generator] ───▶ (Groq LLM Synthesis via llama-3.1)
   │                       └─▶ (Fallback Chain on rate-limit/error)
   ▼
[5: output_formatter] ───▶ (Generates structured final JSON/PDF format)
   │
   ▼
 (End)
```

**State Schema (`AgentState` TypedDict):**
```python
class AgentState(TypedDict):
    input_data: dict
    ml_results: dict
    risk_analysis: str
    rbi_context: str
    draft_report: str
    final_output: dict
    error: str
```

**Workflow Highlights:**
- **Conditional Routing:** Edge logic routes to error-handling or fallback mechanisms if `report_generator` fails or if critical inputs are missing.
- **RAG Pipeline:** Integrates 25 customized RBI regulation chunks. Utilizes cosine similarity via `all-MiniLM-L6-v2` embeddings, dynamically retrieving the top-3 most relevant clauses for the current application.
- **Prompting Strategies:** Evaluates with `temperature=0.3` for consistent, logical reasoning. Enforces strict JSON schema generation from the LLM, backed by a robust fallback chain for improved reliability.

## 📄 Sample Report Output

```json
{
  "borrower_summary": "Applicant is a salaried professional with 5 years history and moderate DTI ratio.",
  "risk_analysis": "ML models indicate a Medium Risk profile (Probability of Default: 18%) largely driven by recent credit utilization.",
  "lending_recommendation": "Approve with conditions.",
  "recommended_action": "Require a 10% higher down payment or collateral adjustment.",
  "regulatory_references": "Complies with RBI Master Circular on Retail Lending Section 4.1 regarding risk-based pricing.",
  "disclaimer": "This is an AI-generated decision support tool; final approval requires human underwriting review."
}
```

## ⚖️ Responsible AI

- **Bias Mitigations:** Implemented SMOTE and `class_weight='balanced'` to prevent model skew against underrepresented demographics. Analyzed feature distributions using IQR to handle outliers securely. Cross-referenced output against RBI fairness code chunks.
- **Limitations:**
  - AI recommendations are advisory and do not replace human underwriting review.
  - The deterministic nature of the dataset does not capture macroeconomic shifts.
  - LLM generations occasionally lack nuance in edge-case regulatory interpretations.
  - Retrieval (RAG) is limited strictly to the 25 predefined RBI regulation chunks.

## 📁 Project Structure

```text
ml_model/
├── data/
│   └── raw_data.csv
├── models/
│   ├── log_reg_smote.pkl
│   └── tree_balanced.pkl
├── agent/
│   ├── graph.py
│   ├── nodes.py
│   └── prompts.py
├── vector_store/
│   └── chroma_db/
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

1. Clone the repository and navigate into the folder.
   `git clone <repo-url> && cd ml_model`
2. Install the required dependencies.
   `pip install -r requirements.txt`
3. Set your API Keys as Environment Variables.
   `export GROQ_API_KEY="your_key"`
4. Run the Streamlit UI.
   `streamlit run app.py`

## 🔗 Deployment

- **Live App:** [Link to Streamlit Cloud / HF Spaces]
- **Demo Video:** [Link to Loom/YouTube]
- **GitHub:** [Link to Repo]

## 👥 Team

- **Kushagra Bhardwaj** — Data preprocessing & feature engineering
- **Vaibhav Singh** — Logistic Regression, evaluation metrics
- **Supreet** — EDA, Streamlit UI
- **Saksham Narotra** — Decision Tree, LangGraph, RAG, LLM
