import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone

# Configuration
folder_path = 'IADemos/processedText'  # Folder path
pinecone_api_key = 'cfda5a85-54d2-487c-95e7-c665d0599dca'  # Pinecone API key
index_name = 'iaindex'  # Pinecone index name
model_name = 'all-MiniLM-L6-v2'  # Sentence transformer model name

# Initialize the sentence transformer model
model = SentenceTransformer(model_name)

def file_to_embedding(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        embedding = model.encode(content)
        return embedding.tolist()  # Convert ndarray to list

def process_folder(folder_path):
    embeddings = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            embedding = file_to_embedding(file_path)
            embeddings.append((filename, embedding))
    return embeddings

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment='gcp-starter')

# Delete the existing index if it exists
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# Create a new index with the correct dimension
embedding_dimension = model.get_sentence_embedding_dimension()
pinecone.create_index(name=index_name, dimension=embedding_dimension)

# Reconnect to the new index
index = pinecone.Index(index_name)

# Process the folder and upload embeddings
embeddings = process_folder(folder_path)
index.upsert(vectors=embeddings)
