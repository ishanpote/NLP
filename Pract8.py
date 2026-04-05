# GROUP 1 - Analyze semantic similarity between sentences using a semantic model or embeddings.

# Semantic Similarity using Embeddings (User Input + Lightweight)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide warnings

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load lightweight model (fast + low memory)
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------- USER INPUT --------
sentence1 = input("Enter first sentence: ")
sentence2 = input("Enter second sentence: ")

# -------- GENERATE EMBEDDINGS --------
embeddings = model.encode([sentence1, sentence2])

# -------- COMPUTE SIMILARITY --------
similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

# -------- OUTPUT --------
print("\nSemantic Similarity Score:", round(similarity, 3))