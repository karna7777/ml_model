import re
query_text = "myself Kamyar with mariied and income of 1000000"
q = query_text.lower()
name_match = re.search(r"(?:name\s*(?:is|:)?|myself|i\s*am|this\s*is)\s*([A-Za-z]+)", query_text, re.IGNORECASE)
extracted_name = name_match.group(1).capitalize() if name_match else "Unknown"
print(f"Extracted Name: {extracted_name}")

def _extract_number(text, keywords, default):
    for kw in keywords:
        pattern = rf"{kw}\s*(?:of|is|:|-)?\s*([\d,.]+)"
        m = re.search(pattern, text)
        if m:
            val_str = m.group(1).replace(",", "")
            if val_str.replace(".", "").isdigit():
                return float(val_str)
        pattern_back = rf"([\d,.]+)\s*(?:years?\s+(?:of\s+)?){kw}"
        m_back = re.search(pattern_back, text)
        if m_back:
            val_str = m_back.group(1).replace(",", "")
            if val_str.replace(".", "").isdigit():
                return float(val_str)
    return default

income = _extract_number(q, ["income", "salary", "earning"], 200000)
print(f"Income: {income}")

family_map = {
    "marr": "Married", "sing": "Single / not married",
    "civil": "Civil marriage", "sep": "Separated", "widow": "Widow",
}
family_q = "Single / not married"
for k, v in family_map.items():
    if k in q:
        family_q = v
        break
print(f"Family Status: {family_q}")
