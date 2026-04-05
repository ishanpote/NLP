# GROUP 1 - Analyze semantic similarity between sentences using a semantic model or embeddings.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

embeddings = model.encode([sentence1, sentence2])

similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

print("\nSemantic Similarity Score:", round(similarity, 3))