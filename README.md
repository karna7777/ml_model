# Project: Credit Risk Scoring & Agentic Lending Decision Support

## From Credit Risk Prediction to Intelligent Lending Assistance

---

## Project Overview
This project focuses on the design and implementation of an **AI-driven credit risk scoring system** that predicts borrower creditworthiness and later evolves into an **agentic AI–based lending decision support assistant**.

- **Milestone 1:** Classical supervised machine learning techniques applied to historical borrower data to predict credit risk and analyze key risk drivers.
- **Milestone 2:** Extension into an agentic AI system that autonomously reasons about borrower risk, retrieves regulatory and lending guidelines (RAG), and generates structured lending recommendations.

The project emphasizes **responsible ML practices**, including data preprocessing, avoidance of data leakage, and appropriate evaluation for imbalanced datasets.

---

## Constraints & Requirements
- **Team Size:** 3–4 Students  
- **API Budget:** Free Tier Only (Open-source models / Free APIs)  
- **Agent Framework (M2):** LangGraph (Recommended)  
- **Hosting:** Mandatory for End-Sem (Hugging Face Spaces, Streamlit Cloud, or Render)

---

## Technology Stack

| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Logistic Regression, Decision Trees (Scikit-Learn) |
| **Agent Framework (M2)** | LangGraph |
| **Vector Store (M2)** | Chroma / FAISS (RAG) |
| **UI Framework** | Streamlit / Gradio |
| **LLMs (M2)** | Open-source models or Free-tier APIs |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib / Seaborn |

---

## Milestones & Deliverables

### Milestone 1: ML-Based Credit Risk Scoring (Mid-Sem)

**Objective:**  
Predict whether a borrower is **risky or non-risky** using historical demographic and financial data, focusing on classical ML pipelines *without using LLMs*.

**Key Deliverables:**
- Problem understanding and lending use-case definition  
- Clean ML pipeline with preprocessing and feature engineering  
- Removal of data leakage features (repayment status, credit timeline)  
- Logistic Regression and Decision Tree models  
- Model evaluation using Accuracy, Confusion Matrix, ROC-AUC  
- Working local application with UI (Streamlit/Gradio)  
- Technical report documenting methodology and results  

---

### Milestone 2: Agentic AI Lending Decision Assistant (End-Sem)

**Objective:**  
Extend the credit risk model into an **agentic AI assistant** that autonomously reasons about borrower risk and generates structured lending recommendations.

**Key Deliverables:**
- **Publicly deployed application** (Link required)  
- Agent workflow documentation (States & Nodes)  
- Retrieval-Augmented Generation (RAG) for lending rules and regulations  
- Structured credit risk and lending recommendation reports  
- Complete GitHub repository with clean commit history  
- Demo video (Maximum 5 minutes)  

---

## Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | ML technique application, Feature Engineering, Data Leakage Handling, UI Usability, Evaluation Metrics |
| **End-Sem** | 30% | Agent reasoning quality, RAG implementation, State management, Output clarity, Deployment success |

> ⚠️ **Important Note**  
> Localhost-only demonstrations will **not** be accepted for final submission.  
> The project must be **publicly hosted** for End-Sem evaluation.

---

## Dataset
- **Source:** Kaggle – Credit Card Approval Dataset  
- **Type:** Structured tabular data  
- **Target Variable:**  
  - `0` → Non-risk borrower  
  - `1` → Risky borrower  

The dataset is **highly imbalanced**, reflecting real-world lending scenarios.

---

## Team Contributions

| Member | Contribution |
|------|-------------|
| Kushagra Bhardwaj | Data preprocessing, feature engineering, leakage handling |
| Vaibhav Singh | Logistic Regression model training, evaluation, report drafting |
| Supreet | Dataset sourcing, EDA, frontend/demo interface |
| Saksham Narotra | Decision Tree model training and performance comparison |

---

## Future Scope
- Integration of agentic AI for autonomous lending decisions  
- Explainable AI using feature importance and rule extraction  
- Deployment with real-time borrower input  
- Regulatory guideline retrieval using RAG  

---

## License
This project is developed for academic purposes as part of the **Intro to GenAI Capstone Project**.
