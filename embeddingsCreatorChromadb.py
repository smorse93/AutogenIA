import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize ChromaDB client and Sentence Transformer
client = chromadb.Client()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory containing your documents
docs_path = '/Users/stevenmorse33/Documents/ResearchStrategy/ProcessedText/'

# Function to create embeddings
def create_embeddings(doc_path):
    with open(doc_path, 'r') as file:
        content = file.read()
    return model.encode(content)

# Create and populate the database
collection_name = 'IATestCollection'
client.create_collection(collection_name)

for filename in os.listdir(docs_path):
    file_path = os.path.join(docs_path, filename)
    if os.path.isfile(file_path):
        embedding = create_embeddings(file_path)
        client.insert(collection_name, embedding, metadata={'filename': filename})
