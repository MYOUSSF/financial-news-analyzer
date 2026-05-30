"""src/utils package"""
from .cache import CacheClient

# VectorStore depends on chromadb which is not always installed (e.g. test env).
# Import it lazily so the rest of src.utils is usable without chromadb.
try:
    from .vector_store import VectorStore
    __all__ = ["CacheClient", "VectorStore"]
except ImportError:
    __all__ = ["CacheClient"]
