import os
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()
dict_path = os.path.join(os.path.dirname(__file__), '..\db\id_rdrsrc\kamus.txt')

with open(dict_path, 'r', encoding='utf-8') as file:
    dictionary = set(line.strip() for line in file if line.strip())

def id_txt_preprocess(tokens):
    case_fold = tokens.lower()
    removed_punct = re.sub(r'[^\w\s-]', '', case_fold)
    words = removed_punct.split()
    prep_tokens = []
    for word in words:
        if word in dictionary:
            stopped = stopword.remove(word)
            if stopped:
                prep_tokens.append(stopped)
        elif re.fullmatch(r'(\w+)-\1', word):
            base = re.sub(r'(\w+)-\1', r'\1', word)
            prep_tokens.append(base)
        else:
            base = stemmer.stem(word)
            prep_tokens.append(base)

    return prep_tokens