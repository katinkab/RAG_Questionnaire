# processing/extractionfrompdf.py 
# extracts all text from pdfs in folder "folder_path" using pymupdf4llm
import os
import re
import pymupdf4llm

def clean_markdown(text):
    text = text.replace('<br>', ' ').replace('<br/>', ' ')
    text = re.sub(r'\[\d+[\s\d]*\]', '', text)
    text = re.sub(r'----- Start of picture text -----', '', text)
    text = re.sub(r'----- End of picture text -----', '', text)
    text = re.sub(r'==> picture.*?<==', '', text)
    text = re.sub(r'■', '-', text)
    
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\|[-\s|]+\|$', stripped):
            continue
        if stripped.startswith('|'):
            if stripped.count('|') > 2:
                continue
            line = re.sub(r'\|', ' ', line)
        clean_lines.append(line)
    
    text = '\n'.join(clean_lines)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def extract_text_from_pdf(filename):
    md_text = pymupdf4llm.to_markdown(filename)
    return clean_markdown(md_text)

def extract_all_pdfs(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            texts[filename] = extract_text_from_pdf(path)
    return texts
