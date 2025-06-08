from collections import Counter, defaultdict
import math

def term_freq(tokens):
    return dict(Counter(tokens))

def show_term_freq(tf):
    for term, freq in tf.items():
        print(f"'{term}': {freq}")

def count_doc_term_freq(all_documents_terms):
    doc_frequency = defaultdict(int)

    for doc_terms in all_documents_terms:
        unique_terms = set(doc_terms)

        for term in unique_terms:
            doc_frequency[term] += 1
    
    return dict(doc_frequency)

def show_doc_term_freq(doc_frequency):
    for term, freq in doc_frequency.items():
        print(f"'{term}': {freq}")

def calc_idf(N, doc_frequency):
    idf_values = {}
    for term, dfi in doc_frequency.items():
        n_dfi = N / dfi
        idf_value = math.log10(n_dfi + 1)
        idf_values[term] = round(idf_value, 4)
    return idf_values

def calc_term_weight(term_freq, idf_values):
    term_weights = {}
    
    for term, tf in term_freq.items():
        if term in idf_values:
            weight = tf * idf_values[term]
            term_weights[term] = round(weight, 4)
    
    return term_weights

def calc_dot_product(q_weights, doc_weights):
    dot_product = 0
    for term in q_weights:
        if term in doc_weights:
            dot_product += q_weights[term] * doc_weights[term]
    return round(dot_product, 4)

def calc_magnitude(weights):
    return round(math.sqrt(sum(w * w for w in weights.values())), 4)

def calc_cosine_similarity(q_weight, doc_weights):
    dot_product = calc_dot_product(q_weight, doc_weights)
    query_magnitude = calc_magnitude(q_weight)
    doc_magnitude = calc_magnitude(doc_weights)
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    similarity = dot_product / (query_magnitude * doc_magnitude)
    return round(similarity, 4)

def calculate_similarities(q_weight, all_documents_weights):
    similarities = []
    for doc_name, doc_weights in all_documents_weights:
        dot_product = calc_dot_product(q_weight, doc_weights)
        query_magnitude = calc_magnitude(q_weight)
        doc_magnitude = calc_magnitude(doc_weights)
        product_magnitude = round(query_magnitude * doc_magnitude, 4)
        similarity = calc_cosine_similarity(q_weight, doc_weights)
        
        similarities.append({
            'document': doc_name,
            'dot_product': dot_product,
            'query_magnitude': query_magnitude,
            'doc_magnitude': doc_magnitude,
            'product_magnitude': product_magnitude,
            'similarity': similarity
        })
    
    return sorted(similarities, key=lambda x: x['similarity'], reverse=True)