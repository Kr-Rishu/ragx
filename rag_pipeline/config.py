import os

EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
EMBEDDING_DIMENSION = 1024
INDEX_BIT_WIDTH = 4

ANSWER_MODEL_NAME = 'gpt-4o-mini'
JUDGE_MODEL_NAME = 'gpt-4o-mini'

# ahttps://openai.com/api/pricing/
ANSWER_INPUT_COST_PER_M_TOKENS = 0.15
ANSWER_OUTPUT_COST_PER_M_TOKENS = 0.60

DEFAULT_TOP_K = int('5')

# s3 Vectors backend - only read if that backend is selected
S3_VECTOR_BUCKET = os.environ.get('RAG_S3_VECTOR_BUCKET')
S3_METADATA_BUCKET = os.environ.get('RAG_S3_METADATA_BUCKET')
AWS_REGION = os.environ.get('REGION')
BEDROCK_EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'
BEDROCK_EMBEDDING_MAX_WORKERS = 5
S3_RETENTION_LIMIT = 3
S3_PUT_VECTORS_BATCH_SIZE = 1000
S3_PROJECT_TAG = 'RAG'

JUDGE_INPUT_COST_PER_M_TOKENS = 0.15
JUDGE_OUTPUT_COST_PER_M_TOKENS = 0.60

INGESTION_ASSET_INPUT_COST_PER_M_TOKENS = 0.15
INGESTION_ASSET_OUTPUT_COST_PER_M_TOKENS = 0.60
INGESTION_BASIC_INPUT_COST_PER_M_TOKENS = 0.15
INGESTION_BASIC_OUTPUT_COST_PER_M_TOKENS = 0.60

BEDROCK_EMBEDDING_COST_PER_M_TOKENS = 0.02

USD_TO_INR_RATE = 94.49

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "what", "which", "who", "how", "does", "do",
    "this", "that", "these", "those", "it", "its", "be", "with", "as",
}
