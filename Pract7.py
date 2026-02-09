# GROUP 1 - Compare POS tagging results between two different NLP systems.

import nltk
import spacy
from nltk import word_tokenize, pos_tag

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

nlp = spacy.load("en_core_web_sm")

sentence = input("Enter a sentence for POS comparison: ")

# -------- NLTK POS Tagging --------
tokens = word_tokenize(sentence)
nltk_tags = pos_tag(tokens)

print("\nNLTK POS Tags:")
for word, tag in nltk_tags:
    print(f"{word} -> {tag}")

# -------- spaCy POS Tagging --------
doc = nlp(sentence)

print("\nspaCy POS Tags:")
for token in doc:
    print(f"{token.text} -> {token.pos_}")