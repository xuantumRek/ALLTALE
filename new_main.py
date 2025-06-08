# new_main.py
import os
import time
from program.load_db import get_docs, read_pdf_file
from program.new_preprocess import id_txt_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    start_time = time.time()

    docs = get_docs()
    doc_names = []
    doc_texts = []

    if not docs:
        print("No documents found.")
        return

    print(f"Processing {len(docs)} documents...")

    for doc in docs:
        if not doc.lower().endswith('.pdf'):
            print("Skipping unsupported file:", doc)
            continue

        try:
            t0 = time.time()
            content = read_pdf_file(doc)
            preprocessed = id_txt_preprocess(content)
            doc_texts.append(preprocessed)
            doc_names.append(os.path.basename(doc))
            print(f"{os.path.basename(doc)} processed in {round(time.time() - t0, 2)}s")
        except Exception as e:
            print(f"Failed to process {doc}: {e}")

    print(f"Total load and preprocess time: {round(time.time() - start_time, 2)} seconds.\n")

    while True:
        query = input("Enter your query (or 'q' to quit): ")
        if query.lower() in ['exit', 'keluar', 'q']:
            break

        preprocessed_query = id_txt_preprocess(query)
        all_texts = [preprocessed_query] + doc_texts

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        ranked_results = sorted(zip(doc_names, similarities), key=lambda x: x[1], reverse=True)

        print("\nTop matched documents:")
        for name, sim in ranked_results:
            print(f"{name} => Cosine Similarity: {round(sim, 4)}")

if __name__ == "__main__":
    main()
