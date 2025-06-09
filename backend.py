# new_main.py
import os
import time
from program.load_db import get_docs, read_pdf_file
from program.new_preprocess import id_txt_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__, template_folder='view')

# Load and preprocess documents once at startup
def load_documents():
    docs = get_docs()
    doc_names = []
    doc_texts = []
    for doc in docs:
        if not doc.lower().endswith('.pdf'):
            continue
        try:
            content = read_pdf_file(doc)
            preprocessed = id_txt_preprocess(content)
            doc_texts.append(preprocessed)
            doc_names.append(os.path.basename(doc))
        except Exception:
            pass
    return doc_names, doc_texts

DOC_NAMES, DOC_TEXTS = load_documents()

def search_documents(query, doc_names, doc_texts):
    preprocessed_query = id_txt_preprocess(query)
    all_texts = [preprocessed_query] + doc_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_results = sorted(zip(doc_names, similarities), key=lambda x: x[1], reverse=True)
    # Ambil hanya dokumen dengan similarity > 0
    filtered = [name for name, sim in ranked_results if sim > 0]
    return filtered

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    kata_kunci = request.form.get('keyword', '')
    if not kata_kunci:
        return render_template('index.html', results=[])
    results = search_documents(kata_kunci, DOC_NAMES, DOC_TEXTS)
    return render_template('index.html', results=results)

@app.route('/download/<filename>')
def download_file(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(base_dir, 'db', 'en_rdsrc'),
        os.path.join(base_dir, 'db', 'id_rdsrc'),
        os.path.join(base_dir, 'db', 'jp_rdsrc'),
    ]
    for directory in search_dirs:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            return send_from_directory(directory, filename, as_attachment=True)
    return "File tidak ditemukan", 404

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')