# GROUP 1 - Compute semantic similarity between two sentences using embeddings or vector-based representations.

"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print("\nSemantic Similarity Score:", similarity[0][0])

"""

# Improved Semantic Similarity using Embeddings + POS awareness

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load lightweight models
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

sentence1 = "I want a book."
sentence2 = "I want to book."

# ---- Embedding Similarity ----
emb1 = model.encode([sentence1])
emb2 = model.encode([sentence2])
semantic_score = cosine_similarity(emb1, emb2)[0][0]

# ---- POS Pattern Comparison ----
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)

pos1 = [token.pos_ for token in doc1]
pos2 = [token.pos_ for token in doc2]

# simple POS similarity
pos_match = sum(p1 == p2 for p1, p2 in zip(pos1, pos2))
pos_score = pos_match / max(len(pos1), len(pos2))

# ---- Final Combined Score ----
final_score = (0.7 * semantic_score) + (0.3 * pos_score)

print("Semantic Score:", semantic_score)
print("POS Score:", pos_score)
print("Final Adjusted Similarity:", final_score)