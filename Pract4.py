# GROUP 1 - Remove stop words using a custom stop-word list created by the student.

import re

STOPWORDS = {
    "is","am","are","was","were","the","a","an","and","or","in","on","at",
    "to","of","for","with","that","this","it","be","by"
}

def remove_stopwords(tokens: list) -> list:
    return [w for w in tokens if w not in STOPWORDS]

def preprocess_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    return remove_stopwords(tokens)

sample = "   AI    is\nTransforming   the WORLD!!!  "
print(preprocess_text(sample))