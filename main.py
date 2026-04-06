from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDF
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

print("Pages loaded:", len(documents))

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print("Total chunks created:", len(docs))

# Print first chunk
print("\nFirst chunk preview:\n")
print(docs[0].page_content)

from sentence_transformers import SentenceTransformer

print("\nCreating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc.page_content for doc in docs]

embeddings = model.encode(texts)

print("Embeddings created:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))
# Embeddings stored in memory
print("Embeddings managed in local memory.")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

query = input("\nAsk a question about the PDF: ")

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)
most_similar_index = np.argmax(similarities)

print("\nMost relevant text:\n")
print(texts[most_similar_index])

context = texts[most_similar_index]


print("\nMost relevant text:\n")
print(context)



from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)


prompt = f"""
You are an assistant answering questions from a document.

Context:
{context}

Question:
{query}

Answer:
"""

result = generator(prompt, max_new_tokens=120)

print("\nAI Answer:\n")
print(result[0]["generated_text"])