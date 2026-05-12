# processing/extractionfrompdf.py 
# extracts all text from pdfs in folder "folder_path"
import os
import pypdfium2 as pdfium

def extract_text_from_pdf(filename):
    pdf = pdfium.PdfDocument(filename)
    text = []
    for page in pdf:
        text_page = page.get_textpage()
        text.append(text_page.get_text_range())  
        text_page.close() 
    pdf.close()
    return "\n\n".join(text)

def extract_all_pdfs(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            texts[filename] = extract_text_from_pdf(path)
    return texts
