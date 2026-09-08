from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME

_local_embedding_model = None

def get_local_embedding_model():
    global _local_embedding_model
    if _local_embedding_model is None:
        _local_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _local_embedding_model
