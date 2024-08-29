from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from google.cloud import aiplatform
import json

# Load environment variables from .env file
load_dotenv()

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

# Get the Pinecone index
index = pc.Index("rmp-rag")

# Load the review data
with open("reviews.json", "r") as file:
    data = json.load(file)

# Extract the list of reviews from the JSON structure
reviews = data.get("reviews", [])

# Ensure 'reviews' is a list and print the first two items to check structure
if isinstance(reviews, list):
    print("First two reviews:", reviews[:2])
else:
    raise ValueError("The 'reviews' key is not a list. Check your JSON structure.")

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/snith/Downloads/rmp.json"

# Initialize Google AI Studio (Vertex AI)
aiplatform.init(project='rate-my-professor-433823', location='us-central1')

# Define model and endpoint details
model_id = "textembedding-gecko"  # Replace with your actual model ID
endpoint_display_name = "rmp-text-embedding-endpoint"  # Your chosen name for the endpoint

# Get the model resource by its ID
model = aiplatform.Model(model_name=model_id)

# Deploy the model to a new endpoint
endpoint = model.deploy(
    endpoint_display_name=endpoint_display_name,
    machine_type="n1-standard-4",  # Choose a suitable machine type
    min_replica_count=1,
    max_replica_count=1,
)

print(f"Model deployed to endpoint: {endpoint.resource_name}")

# Print the Endpoint ID for use in predictions
print(f"Endpoint ID: {endpoint.resource_name}")


# Function to create embeddings for each review
def create_embeddings(text):
    instances = [{"content": text}]
    response = endpoint.predict(instances=instances)
    return response.predictions[0]  # Extract the embeddings from the response

# Process each review and create embeddings
for i, review in enumerate(reviews):
    # Extract the review text from each review object
    review_text = review.get('review', '')
    if not review_text:
        print(f"Skipping review {i+1} because it has no text.")
        continue
    
    # Create embeddings for the review text
    embedding = create_embeddings(review_text)
    print(f"Review {i+1} Embedding: {embedding}")

    # Store embedding in Pinecone
    index.upsert([
        {"id": f"review-{i}", "values": embedding}
    ])
    print(f"Review {i+1} stored in Pinecone.")
