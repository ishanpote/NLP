# GROUP 1 - Compute semantic similarity between two sentences using embeddings or vector-based representations.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print("\nSemantic Similarity Score:", similarity[0][0])