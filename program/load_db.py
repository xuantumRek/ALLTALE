# program/load_db.py
import os
import pdfplumber
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def get_docs():
    dir_path = os.path.join(os.path.dirname(__file__), '../db/id_rdrsrc')
    docs_list = []
    allowed_extensions = ('.pdf', )
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(allowed_extensions):
            docs_list.append(os.path.join(dir_path, filename))
    return docs_list

def read_pdf_file(file):
    base_name = os.path.basename(file)  # misal: 978-623-00-4912-5.pdf
    pdf_dir = os.path.dirname(file)     # direktori PDF

    # Simpan cache di direktori yang sama dengan file PDF
    cache_file = os.path.join(pdf_dir, base_name + ".txt")

    if os.path.exists(cache_file):
        print(f"[CACHE] Loaded cached text for {base_name}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()

    try:
        content = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"

        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[CACHE] Saved new cache for {base_name}")
        return content
    except Exception as e:
        print(f"[ERROR] Failed to read {base_name}: {e}")
        return ""

