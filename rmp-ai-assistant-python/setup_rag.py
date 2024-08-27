from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from google.cloud import aiplatform as genai
import os
import json

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index already exists
existing_indexes = pc.list_indexes().names()
print("Existing indexes: ", existing_indexes)

# Create a Pinecone index if it doesn't exist
if "rmp-rag" not in existing_indexes:
    pc.create_index(
        name="rmp-rag",
        dimension=768,  # Gemini uses 768-dimensional embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index 'rmp-rag' created successfully.")
else:
    print("Index 'rmp-rag' already exists. Skipping creation.")

# Get the index
index = pc.Index("rmp-rag")

# Load the review data
with open("reviews.json", "r") as f:
    data = json.load(f)

processed_data = []

# Initialize Google AI Studio
genai.init(api_key=os.getenv("GOOGLE_API_KEY"))

# Create embeddings for each review
for review in data["reviews"]:
    embedding = genai.TextEmbedding(model="models/embedding-001").embed_text(
        content=review['review']
    )["embedding"]
    
    processed_data.append(
        {
            "values": embedding,
            "id": review["professor"],
            "metadata": {
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())
