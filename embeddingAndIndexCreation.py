import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone

# Configuration
folder_path = '/Users/stevenmorse33/Documents/ResearchStrategy/ProcessedText'  # Folder path
pinecone_api_key = '370c4b4d-ad85-4463-a7ce-097d78a3dfbb'  # Pinecone API key
index_name = 'iaindex'  # Pinecone index name
model_name = 'all-MiniLM-L6-v2'  # Sentence transformer model name

# Initialize the sentence transformer model
model = SentenceTransformer(model_name)

def file_to_embedding(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # Try a different encoding or ignore the error
        with open(file_path, 'r', encoding='latin-1') as file:  # Fallback encoding
            content = file.read()

    embedding = model.encode(content)
    return embedding.tolist()

def process_folder(folder_path):
    embeddings = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):  # Process only .txt files
            continue
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                embedding = file_to_embedding(file_path)
                embeddings.append((filename, embedding))
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
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
