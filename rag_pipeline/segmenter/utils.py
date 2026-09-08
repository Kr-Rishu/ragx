import nltk
import time
import hashlib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')

NLTK_DATA_DIR = "./nltk_data"
nltk.data.path.append(NLTK_DATA_DIR)

def generate_sha256_hash(content):
    bytes = content.encode()
    hashobj = hashlib.sha256(bytes)
    return hashobj.hexdigest()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    encoded_text = text.lower().encode('ascii', 'ignore').decode()
    tokens = word_tokenize(encoded_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def create_summary_chunks(chunks, fixed_keys):
    new_chunks = []
    for chunk in chunks:
        new_chunk = {}
        for k, v in chunk.items():
            if k in fixed_keys:
                new_chunk[k] = v
            elif k == "summary":
                new_chunk["embedding_text"] = v
        new_chunks.append(new_chunk)
    return new_chunks

def create_questions_chunks(chunks, fixed_keys):
    new_chunks = []
    for chunk in chunks:
        new_chunk = {}
        for k, v in chunk.items():
            if k in fixed_keys:
                new_chunk[k] = v
            elif k == "questions":
                new_chunk["embedding_text"] = "\n".join(v)
        new_chunks.append(new_chunk)
    return new_chunks

def get_file_hash(file_path):
    print('Generating file hash')

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    file_hash = hasher.hexdigest()
    print(f'File hash: {file_hash}')

    return file_hash

class Timer:
    
    def __init__(self, process_name : str):
        self.process_name = process_name
        
    def __enter__(self):
        self.start_time = time.time()
        
    def __exit__(self, exc_type, exc, tb) -> None:
        
        self.end_time = time.time()
        if not exc_type:
            
            print(f'[+] {self.process_name} took - {round(self.end_time-self.start_time,3)}s')
            return  
