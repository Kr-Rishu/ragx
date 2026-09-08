from .base import VectorStorageBackend
from .turbovec_sqlite import TurbovecSqliteBackend
from .s3_vectors import S3VectorsBackend

def get_storage_backend(backend_name, collection_name, **kwargs):
    if backend_name == 'turbovec_sqlite':
        collection_dir = kwargs.get('collection_dir', './rag_collections')
        force_rebuild = kwargs.get('force_rebuild', False)
        return TurbovecSqliteBackend(collection_dir=collection_dir, collection_name=collection_name, force_rebuild=force_rebuild)
    
    elif backend_name == 's3_vectors':
        kwargs.pop('collection_dir', None)
        kwargs.pop('force_rebuild', None)
        return S3VectorsBackend(collection_name=collection_name, **kwargs)
    
    raise ValueError(f'Unknown storage backend: {backend_name}')
