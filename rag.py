import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# Load JSON
# -----------------------------
with open("portfolio_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON to text
documents = []

for key, value in data.items():
    documents.append(f"{key}: {value}")

# -----------------------------
# Embedding Model
# -----------------------------
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Create embeddings
embeddings = embedding_model.encode(documents)

# -----------------------------
# FAISS Index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

# -----------------------------
# Retriever
# -----------------------------
def retrieve(query, k=10):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    retrieved_docs = []

    for idx in indices[0]:
        retrieved_docs.append(documents[idx])

    return "\n".join(retrieved_docs)

# -----------------------------
# LLM
# -----------------------------
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def ask_portfolio(question):

    context = retrieve(question)

    prompt = f"""
You are an assistant answering questions about my portfolio.

Context:
{context}

Question:
{question}

Answer only from the context.
If answer is not available, say:
"I could not find that information in the portfolio."
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# # -----------------------------
# # Chat Loop
# # -----------------------------
# while True:

#     query = input("\nAsk: ")

#     if query.lower() == "exit":
#         break

#     answer = ask_portfolio(query)

    # return answer