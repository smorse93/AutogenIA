import os
from tika import parser
#from tika import config
#config.set_service('/Users/stevenmorse33/Documents/Coding Projects/MicrosoftAutogenNotebooks/IADemos/dependencies/tika-server-standard-2.9.1.jar')
import re
import nltk

# Download NLTK stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    # Substitute multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

output_dir = 'IADemos/processedText'
os.makedirs(output_dir, exist_ok=True)  # This ensures that the directory exists

def convert_folder_to_text(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if it's a file
        if os.path.isfile(file_path):
            # Parse the file with Apache Tika
            parsed = parser.from_file(file_path)
            # Preprocess the text
            processed_text = preprocess_text(parsed["content"])
            # Define the output file path
            output_file_path = os.path.join(output_dir, f"{filename}_processed.txt")
            # Save the processed text
            with open(output_file_path, "w") as text_file:
                text_file.write(processed_text)
            print(f"Processed: {filename}")


# Replace 'your_folder_path' with the path to the folder containing your documents
convert_folder_to_text('/Users/stevenmorse33/Documents/Coding Projects/MicrosoftAutogenNotebooks/IADemos/rag_data')
