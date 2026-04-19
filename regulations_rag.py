"""
regulations_rag.py
──────────────────
RAG (Retrieval-Augmented Generation) module for the Credit Card Approval
Lending Decision Support system.

Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Contains 25 realistic RBI / lending regulation guideline chunks.
"""

import os
import chromadb
from chromadb.config import Settings

# ────────────────────────────────────────────────────────────────
# 25 Regulation / Guideline chunks (RBI-style lending rules)
# ────────────────────────────────────────────────────────────────

REGULATION_CHUNKS = [
    # 1
    (
        "reg_001",
        "RBI Guidelines on Minimum Creditworthiness Threshold: "
        "All regulated lending entities must assess the creditworthiness of a borrower "
        "before sanctioning any credit facility. A minimum credit score of 650 on the "
        "CIBIL scale (or equivalent on other bureaus) is recommended for unsecured retail "
        "credit products. Borrowers falling below this threshold must be subject to "
        "enhanced due diligence and higher provisioning norms."
    ),
    # 2
    (
        "reg_002",
        "Income-to-Loan Ratio Limits: "
        "The Equated Monthly Installment (EMI) for any new credit facility should not "
        "exceed 40%% of the borrower's net monthly income. Lenders must verify income "
        "through salary slips, bank statements, or IT returns for the preceding two "
        "financial years. Any deviation must be approved by the credit committee and "
        "documented with justification."
    ),
    # 3
    (
        "reg_003",
        "Employment Stability Requirement: "
        "For salaried applicants, a minimum of 2 years of continuous employment is "
        "preferred. Frequent job changes (more than 3 employers in 2 years) may indicate "
        "income instability and should be flagged for additional review. Employment tenure "
        "is a key factor in assessing repayment capacity."
    ),
    # 4
    (
        "reg_004",
        "Self-Employed Borrower Documentation: "
        "Self-employed individuals must furnish Income Tax Returns (ITR) for the "
        "preceding 3 financial years, along with audited financial statements and bank "
        "account statements for the last 12 months. The average of the last 3 years' net "
        "income shall be used for eligibility computation."
    ),
    # 5
    (
        "reg_005",
        "Age Criteria for Credit Facility: "
        "The minimum age for availing a credit card or personal loan is 21 years. "
        "For salaried individuals, the maximum age at loan maturity should not exceed "
        "60 years (retirement age). For self-employed individuals, the upper limit is "
        "65 years at loan maturity. Age is used as a proxy for earning horizon and "
        "repayment capacity."
    ),
    # 6
    (
        "reg_006",
        "Credit History — Default Disqualification: "
        "Any borrower with a recorded default, write-off, or settlement in the past "
        "24 months is generally ineligible for new unsecured credit. The lender must "
        "verify the borrower's credit history from at least one RBI-recognized credit "
        "bureau. Accounts with DPD (Days Past Due) greater than 90 days in the last "
        "2 years constitute adverse credit history."
    ),
    # 7
    (
        "reg_007",
        "CIBIL Score Threshold for Standard Loans: "
        "A CIBIL score of 650 or above is the minimum benchmark for standard retail "
        "credit products including credit cards, personal loans, and consumer durable "
        "loans. Scores between 600 and 650 may be considered under a higher interest "
        "rate tier. Scores below 600 generally warrant rejection unless supported by "
        "collateral or a co-applicant with adequate score."
    ),
    # 8
    (
        "reg_008",
        "Debt-to-Income Ratio Limit: "
        "The total monthly debt obligations of the borrower, including existing EMIs, "
        "credit card minimum payments, and proposed new EMI, must not exceed 50%% of "
        "the gross monthly income. This ratio is a critical measure of the borrower's "
        "financial leverage and repayment sustainability."
    ),
    # 9
    (
        "reg_009",
        "Co-Applicant Rules for Joint Applications: "
        "Joint applications are permitted where the co-applicant is a spouse, parent, "
        "or sibling. The combined income of both applicants may be considered for "
        "eligibility computation. Both applicants share equal liability for repayment. "
        "Each applicant's credit history must be evaluated independently."
    ),
    # 10
    (
        "reg_010",
        "Housing Ownership as Positive Credit Signal: "
        "Borrowers who own residential property (house or apartment) demonstrate higher "
        "financial stability and are considered lower risk. Property ownership is a "
        "positive factor in credit scoring models and may qualify the borrower for "
        "preferential interest rates."
    ),
    # 11
    (
        "reg_011",
        "Number of Dependents Impact on Repayment Capacity: "
        "The number of financial dependents (children, elderly parents) directly impacts "
        "the borrower's disposable income and repayment capacity. Lenders should factor "
        "in the number of dependents when calculating the adjusted net income available "
        "for debt servicing."
    ),
    # 12
    (
        "reg_012",
        "Gender-Neutral Lending Mandate (RBI Fair Lending Code): "
        "As per RBI's Fair Practices Code, lenders must ensure that credit decisions are "
        "gender-neutral. No applicant shall be offered differential terms or denied credit "
        "solely based on gender. All lending criteria must be applied uniformly across "
        "all applicants regardless of gender identity."
    ),
    # 13
    (
        "reg_013",
        "Education Level as Proxy for Income Stability: "
        "Higher educational qualifications (graduate degree and above) are correlated with "
        "more stable income trajectories and lower default probability. While education "
        "alone should not be a deciding factor, it may be used as one of several inputs "
        "in the credit scoring model along with employment history and income."
    ),
    # 14
    (
        "reg_014",
        "Family Status and Financial Stability: "
        "Married applicants or applicants with stable family units may demonstrate "
        "higher financial responsibility. However, family status must not be used as "
        "a sole criterion for credit decisions. It may serve as supplementary "
        "information in conjunction with income, employment, and credit history data."
    ),
    # 15
    (
        "reg_015",
        "Anti-Discrimination Clause — RBI Fair Lending: "
        "No borrower shall be denied credit or offered inferior terms on the basis of "
        "gender, caste, religion, ethnicity, disability, or any other protected "
        "characteristic. All credit decisions must be based on objective, quantifiable "
        "financial criteria. Any violation is subject to regulatory penalty."
    ),
    # 16
    (
        "reg_016",
        "Responsible AI in Credit Decisions — Explainability Requirement: "
        "Where AI or ML models are used in credit underwriting, the lender must ensure "
        "that the model's decisions are explainable. Borrowers have the right to "
        "understand the key factors that influenced the credit decision. Black-box models "
        "without interpretability layers are discouraged by the RBI for retail lending."
    ),
    # 17
    (
        "reg_017",
        "Data Privacy in Credit Scoring — RBI IT Act Compliance: "
        "All personal and financial data collected for credit assessment must be handled "
        "in compliance with the Information Technology Act, 2000, and RBI's guidelines on "
        "data protection. Data must be stored securely, used only for the stated purpose, "
        "and not shared with third parties without the borrower's explicit consent."
    ),
    # 18
    (
        "reg_018",
        "Maximum Loan-to-Value Ratio for Secured Credit: "
        "For secured credit facilities (e.g., home loans, vehicle loans), the loan amount "
        "should not exceed the prescribed Loan-to-Value (LTV) ratio. For housing loans, "
        "the maximum LTV is 90%% for loans up to Rs 30 lakh, 80%% for loans between "
        "Rs 30-75 lakh, and 75%% for loans above Rs 75 lakh."
    ),
    # 19
    (
        "reg_019",
        "Regulatory Requirement for Rejection Reason Disclosure: "
        "When a credit application is rejected, the lender must provide the borrower with "
        "specific reasons for the rejection in writing within 7 working days. Generic "
        "rejection notices without specific reasons are not compliant with RBI's Fair "
        "Practices Code. This includes adverse action notices referencing credit bureau data."
    ),
    # 20
    (
        "reg_020",
        "Periodic Credit Review Requirements: "
        "For existing credit card and revolving credit customers, the lender must conduct "
        "periodic credit reviews at least once every 12 months. This includes reassessing "
        "the borrower's credit score, income changes, and overall debt position to "
        "determine ongoing eligibility and credit limit adjustments."
    ),
    # 21
    (
        "reg_021",
        "Microfinance Borrower Protections: "
        "Microfinance borrowers are entitled to additional protections including cap on "
        "total indebtedness (maximum 2 MFI lenders), cap on repayment obligation (50%% of "
        "household income), and mandatory cooling-off period between loans. Aggressive "
        "recovery practices are strictly prohibited."
    ),
    # 22
    (
        "reg_022",
        "NPA Classification Rules: "
        "A credit card account is classified as Non-Performing Asset (NPA) when the "
        "minimum amount due remains unpaid for more than 90 days from the payment due "
        "date. Once classified as NPA, the account ceases to generate interest income "
        "for the lender and must be provisioned as per RBI norms."
    ),
    # 23
    (
        "reg_023",
        "Digital Lending Guidelines — RBI 2022 Circular: "
        "All digital lending activities must comply with the RBI's Digital Lending "
        "Guidelines (Sept 2022). Key requirements include: disclosure of all fees upfront, "
        "mandatory cooling-off period for loan cancellation, prohibition on automatic "
        "credit limit increases without consent, and direct fund disbursement to "
        "borrower's bank account only."
    ),
    # 24
    (
        "reg_024",
        "Grievance Redressal Mechanism: "
        "Every lender must establish a multi-tier grievance redressal mechanism accessible "
        "to all borrowers. This includes a dedicated nodal officer, escalation to the "
        "internal ombudsman, and final recourse to the RBI Integrated Ombudsman Scheme. "
        "Complaints must be acknowledged within 3 working days and resolved within 30 days."
    ),
    # 25
    (
        "reg_025",
        "Environmental and Social Risk in Lending (ESG): "
        "Lenders are encouraged to integrate Environmental, Social, and Governance (ESG) "
        "factors into their lending framework. This includes assessing the borrower's "
        "industry for environmental risk, ensuring social responsibility in lending "
        "practices, and maintaining high governance standards in credit committees."
    ),
]


# ────────────────────────────────────────────────────────────────
# ChromaDB setup (persistent storage)
# ────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "lending_regulations"


def initialize_rag():
    """
    Build and persist the ChromaDB collection with regulation chunks.
    Idempotent — skips if the collection already has the expected documents.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Try to get the existing collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        # If collection already has all docs, skip
        if collection.count() >= len(REGULATION_CHUNKS):
            return collection
        else:
            # Collection exists but is incomplete — delete and rebuild
            client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet

    # Create the collection using Chroma's default embedding function
    # (Chroma uses all-MiniLM-L6-v2 by default)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add all regulation chunks
    ids = [chunk[0] for chunk in REGULATION_CHUNKS]
    documents = [chunk[1] for chunk in REGULATION_CHUNKS]
    metadatas = [{"source": "RBI_Guidelines", "chunk_id": chunk[0]} for chunk in REGULATION_CHUNKS]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def retrieve_regulations(query: str, n_results: int = 3) -> list:
    """
    Retrieve the top-n most relevant regulation texts for a given query.

    Parameters
    ----------
    query : str
        The search query (e.g., risk class + risk driver info).
    n_results : int
        Number of regulation chunks to return.

    Returns
    -------
    list[str]
        List of regulation text strings.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        # Collection not initialized yet — do it now
        collection = initialize_rag()

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    if results and results.get("documents") and len(results["documents"]) > 0:
        return results["documents"][0]

    return []


# ────────────────────────────────────────────────────────────────
# Quick self‑test
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing RAG system …")
    col = initialize_rag()
    print(f"Collection created with {col.count()} documents.")

    test_query = "What is the minimum credit score for a loan?"
    docs = retrieve_regulations(test_query, n_results=3)
    print(f"\nQuery: {test_query}")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print(doc[:200] + "…")
