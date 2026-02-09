#GROUP 1 - Design a complete text preprocessing pipeline including whitespace normalization and tokenization.

import re

STOPWORDS = {
    "is","am","are","was","were","the","a","an","and","or","in","on","at","to",
    "of","for","with","that","this","it","be","by","as","from","but"
}

def preprocess_text(text: str) -> list:
    text = text.lower()

    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()

    return [w for w in tokens if w not in STOPWORDS]
sample = "   AI    is\nTransforming   the WORLD!!!  "
print(preprocess_text(sample))