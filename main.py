# main.py
import os
from program.preprocess import id_txt_preprocess
from program.process import term_freq, count_doc_term_freq, calc_idf, calc_term_weight, calculate_similarities
from program.load_db import get_docs, read_pdf_file

def main():
    while True:
        # Query Processing
        query = input("Enter your query: ")
        if query.lower() in ['exit', 'keluar', 'q']:
            print("Program dihentikan.")
            break

        prep_q = id_txt_preprocess(query)
        q_tf = term_freq(prep_q)

        docs = get_docs()
        N = len(docs)
        print(f"\n{N} items founded")
        all_documents_terms = []
        all_documents_weights = []
        
        if docs:
            # print(f"\nDocuments found: ")
            for doc in docs:
                if doc.lower().endswith('.pdf'):
                    content = read_pdf_file(doc)
                else:
                    print("Unsupported file type")
                    continue

                prep_content = id_txt_preprocess(content)
                all_documents_terms.append(prep_content)
                tf_content = term_freq(prep_content)
                
                # print(f"\nTerm frequency for {doc_name}:")
                # show_term_freq(tf_content)

            doc_freq = count_doc_term_freq(all_documents_terms)
            # print("\nDocument frequency for each term:")
            # show_doc_term_freq(doc_freq)

            idf_values = calc_idf(N, doc_freq)
            q_weights = calc_term_weight(q_tf, idf_values)

            for doc in docs:
                doc_name = os.path.basename(doc)
                if doc.lower().endswith('.pdf'):
                    content = read_pdf_file(doc)
                else:
                    print("Unsupported file type")
                    continue

                prep_content = id_txt_preprocess(content)
                all_documents_terms.append(prep_content)
                tf_content = term_freq(prep_content)
                doc_weights = calc_term_weight(tf_content, idf_values)
                all_documents_weights.append((doc_name, doc_weights))

            similarities = calculate_similarities(q_weights, all_documents_weights)
            for sim in similarities:
                print(f"\nDocument: {sim['document']}")
                # print(f"Dot Product: {sim['dot_product']}")
                # print(f"Query Magnitude: {sim['query_magnitude']}")
                # print(f"Document Magnitude: {sim['doc_magnitude']}")
                # print(f"||Q||.||Di||: {sim['product_magnitude']}")
                print(f"Cosine Similarity: {sim['similarity']}")
        else:
            print("No documents found.")

if __name__ == "__main__":
    main()
    