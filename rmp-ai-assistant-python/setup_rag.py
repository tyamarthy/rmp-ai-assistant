from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import os
import json

load_dotenv('.env.local')

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index already exists
existing_indexes = pc.list_indexes()
print("Existing indexes: ", existing_indexes)

# Extract index names from the structure
index_names = [index.name for index in existing_indexes]
print("Index Names: ",index_names)

# Create a Pinecone index if it doesn't exist
if "rmp-rag" not in index_names:
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
data = json.load(open("reviews.json"))

processed_data = []

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create embeddings for each review
for review in data["reviews"]:
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=review['review'],
        task_type="retrieval_document"
    )["embedding"]
    
    processed_data.append(
        {
            "values": embedding,
            "id": review["professor"],
            "metadata":{
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
index = pc.Index("rmp-rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())