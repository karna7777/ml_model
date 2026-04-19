import json
from fpdf import FPDF
from datetime import datetime

def create_pdf_report(profile, risk_score, risk_class, risk_drivers, regs, report):
    def sanitize(text):
        if text is None:
            return "N/A"
        return str(text).replace("₹", "Rs. ").encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font('helvetica', 'B', 20)
    pdf.cell(0, 15, 'AI Lending Assessment Report', border=0, ln=1, align='C')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', border=0, ln=1, align='C')
    pdf.ln(5)
    
    # Proposal Status
    rec = str(report.get("lending_recommendation", "")).upper()
    pdf.set_font('helvetica', 'B', 20)
    if "APPROVE" in rec:
        pdf.set_text_color(34, 197, 94) # Green
        pdf.cell(0, 20, 'STATUS: PROPOSAL ACCEPTED', border=0, ln=1, align='C')
    elif "REJECT" in rec:
        pdf.set_text_color(239, 68, 68) # Red
        pdf.cell(0, 20, 'STATUS: PROPOSAL REJECTED', border=0, ln=1, align='C')
    else:
        pdf.set_text_color(245, 158, 11) # Orange
        pdf.cell(0, 20, 'STATUS: CONDITIONAL APPROVAL', border=0, ln=1, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Section 1: Borrower Profile
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, '1. Borrower Profile', border=0, ln=1)
    pdf.set_font('helvetica', '', 11)
    for k, v in profile.items():
        pdf.cell(0, 6, sanitize(f"{k}: {v}"), border=0, ln=1)
    pdf.ln(5)
    
    # Section 2: AI Risk Assessment
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, '2. AI Risk Assessment', border=0, ln=1)
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 6, sanitize(f"Risk Score: {risk_score:.4f}"), border=0, ln=1)
    pdf.cell(0, 6, sanitize(f"Risk Class: {risk_class}"), border=0, ln=1)
    pdf.cell(0, 6, sanitize(f"Key Risk Drivers: {', '.join(risk_drivers)}"), border=0, ln=1)
    pdf.ln(5)
    
    # Section 3: LLM Analysis
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, '3. AI Summary & Analysis', border=0, ln=1)
    
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 6, 'Summary:', border=0, ln=1)
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 6, sanitize(report.get("borrower_summary", "N/A")))
    pdf.ln(2)
    
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 6, 'Risk Analysis:', border=0, ln=1)
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 6, sanitize(report.get("risk_analysis", "N/A")))
    pdf.ln(2)
    
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 6, 'Recommended Action:', border=0, ln=1)
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 6, sanitize(report.get("recommended_action", "N/A")))
    pdf.ln(5)
    
    # Section 4: Regulatory Context
    if regs:
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, '4. Regulatory Context (RBI Guidelines)', border=0, ln=1)
        pdf.set_font('helvetica', '', 11)
        for i, reg in enumerate(regs, 1):
            pdf.multi_cell(0, 6, sanitize(f"[{i}] {reg}"))
            pdf.ln(2)
        pdf.ln(3)
        
    # Disclaimer
    pdf.set_font('helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, sanitize(f"Disclaimer: {report.get('disclaimer', 'N/A')}"))
    
    return bytes(pdf.output())
