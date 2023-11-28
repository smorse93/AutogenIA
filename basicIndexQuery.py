from sentence_transformers import SentenceTransformer
import pinecone
import os

# Configuration
pinecone_api_key = 'cfda5a85-54d2-487c-95e7-c665d0599dca'  # Pinecone API key
index_name = 'iaindex'  # Pinecone index name
model_name = 'all-MiniLM-L6-v2'  # Sentence transformer model name

# Initialize Pinecone and the Sentence Transformer Model
pinecone.init(api_key=pinecone_api_key, environment='gcp-starter')
model = SentenceTransformer(model_name)
index = pinecone.Index(index_name)

def query_index(query_text, top_k=5):
    # Convert query to embedding
    query_embedding = model.encode(query_text).tolist()

    # Updated search in Pinecone index
    query_results = index.query(query_embedding, top_k=top_k)
    
    # Retrieve and process results
    for match in query_results['matches']:
        doc_id = match['id']
        print(f"Matched document: {doc_id}, Score: {match['score']}")


# Example query
query_text = "What habits do athletes have that were interviewed "
query_index(query_text)
