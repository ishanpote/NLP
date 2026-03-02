# GROUP 1 - Compute semantic similarity between two sentences using embeddings or vector-based representations.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print("\nSemantic Similarity Score:", similarity[0][0])


"""
# Semantic Similarity with User Input
# Embedding-based + lightweight POS adjustment

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide TF warnings

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load lightweight models
model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# -------- USER INPUT --------
sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

# -------- EMBEDDING SIMILARITY --------
embeddings = model.encode([sentence1, sentence2])
semantic_score = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

# -------- POS-BASED PENALTY (fix for noun/verb issue) --------
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)

pos_dict1 = {token.text.lower(): token.pos_ for token in doc1}
pos_dict2 = {token.text.lower(): token.pos_ for token in doc2}

penalty = 0

for word in pos_dict1:
    if word in pos_dict2:
        if pos_dict1[word] != pos_dict2[word]:
            penalty += 0.2   # reduce score if same word has different role

# Final adjusted similarity
final_score = max(0, semantic_score - penalty)

# -------- OUTPUT --------
print("\nSemantic Score :", round(semantic_score, 3))
print("Penalty Applied:", penalty)
print("Final Similarity:", round(final_score, 3))"""