"""
Vector Store Utility - ChromaDB wrapper for storing and retrieving financial news.
"""
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class VectorStore:
    """
    ChromaDB-based vector store for financial news articles and market events.

    Supports:
    - Adding documents with metadata
    - Semantic similarity search
    - Filtered queries by symbol, date, sentiment
    - Collection management
    """

    DEFAULT_COLLECTION = "financial_news"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        """
        Initialize the VectorStore.

        Args:
            persist_directory: Path to persist ChromaDB data. Reads from
                               CHROMA_DB_PATH env var if not provided.
            collection_name: Name of the ChromaDB collection to use.
        """
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_DB_PATH", "./data/chroma_db"
        )
        self.collection_name = collection_name

        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize ChromaDB client (persistent)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Set up embedding function (sentence-transformers, local, no API key needed)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBEDDING_MODEL
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"VectorStore initialized — collection: '{self.collection_name}', "
            f"path: '{self.persist_directory}', "
            f"existing docs: {self.collection.count()}"
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Add a single document to the vector store.

        Args:
            text: Document text to embed and store.
            metadata: Optional metadata dict (symbol, source, date, sentiment, …).
            doc_id: Optional explicit ID; auto-generated if omitted.

        Returns:
            The document ID used.
        """
        doc_id = doc_id or str(uuid.uuid4())
        metadata = self._sanitize_metadata(metadata or {})

        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.debug(f"Added document {doc_id} to '{self.collection_name}'")
        return doc_id

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add multiple documents in a single batch.

        Args:
            texts: List of document texts.
            metadatas: Parallel list of metadata dicts (optional).
            ids: Parallel list of IDs (optional, auto-generated if omitted).

        Returns:
            List of document IDs used.
        """
        if not texts:
            return []

        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = [self._sanitize_metadata(m or {}) for m in (metadatas or [{}] * len(texts))]

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Batch-added {len(texts)} documents to '{self.collection_name}'")
        return ids

    def upsert_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Insert or update a document (idempotent on doc_id).

        Args:
            text: Document text.
            metadata: Optional metadata dict.
            doc_id: Document ID; auto-generated if omitted.

        Returns:
            The document ID used.
        """
        doc_id = doc_id or str(uuid.uuid4())
        metadata = self._sanitize_metadata(metadata or {})

        self.collection.upsert(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.debug(f"Upserted document {doc_id}")
        return doc_id

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search over stored documents.

        Args:
            query: Natural-language query string.
            n_results: Maximum number of results to return.
            where: Optional ChromaDB metadata filter dict.
                   Example: {"symbol": "AAPL"} or {"$and": [{"symbol": "AAPL"}, {"sentiment": "Positive"}]}
            threshold: Minimum similarity score (0–1, cosine). Results below
                       this score are filtered out. 0.0 = no filter.

        Returns:
            List of result dicts with keys: id, document, metadata, score.
        """
        count = self.collection.count()
        if count == 0:
            logger.warning("VectorStore is empty — no results to return.")
            return []

        # ChromaDB raises if n_results > count
        n_results = min(n_results, count)

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        results = []
        for doc, meta, dist, doc_id in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
            raw["ids"][0],
        ):
            # ChromaDB cosine distance → similarity: score = 1 - distance
            score = 1.0 - dist
            if score >= threshold:
                results.append(
                    {
                        "id": doc_id,
                        "document": doc,
                        "metadata": meta,
                        "score": round(score, 4),
                    }
                )

        logger.debug(f"Search '{query[:60]}…' → {len(results)} results (threshold={threshold})")
        return results

    def search_by_symbol(
        self,
        symbol: str,
        query: str = "",
        n_results: int = 10,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search documents filtered to a specific stock symbol.

        Args:
            symbol: Stock ticker (e.g. "AAPL").
            query: Optional semantic query; uses symbol name if empty.
            n_results: Max results.
            threshold: Minimum similarity score.

        Returns:
            Filtered search results.
        """
        query = query or f"{symbol} financial news"
        return self.search(
            query=query,
            n_results=n_results,
            where={"symbol": symbol.upper()},
            threshold=threshold,
        )

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Dict with id, document, metadata, or None if not found.
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            logger.error(f"Error fetching document {doc_id}: {e}")
        return None

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete.

        Returns:
            True if deleted, False on error.
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def delete_by_symbol(self, symbol: str) -> int:
        """
        Delete all documents associated with a stock symbol.

        Args:
            symbol: Stock ticker.

        Returns:
            Number of documents deleted.
        """
        try:
            results = self.collection.get(where={"symbol": symbol.upper()})
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents for symbol {symbol}")
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting documents for {symbol}: {e}")
            return 0

    def clear_collection(self) -> None:
        """Delete all documents in the collection (non-destructive: keeps the collection)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Cleared all documents from collection '{self.collection_name}'")

    # ------------------------------------------------------------------
    # Utility / info
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of documents in the collection."""
        return self.collection.count()

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": self.collection.count(),
            "embedding_model": self.EMBEDDING_MODEL,
        }

    def add_news_article(
        self,
        title: str,
        content: str,
        symbol: str,
        source: str,
        published_at: str,
        sentiment: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        url: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Convenience method: add a financial news article with standard metadata.

        Args:
            title: Article headline.
            content: Article body / summary.
            symbol: Related stock ticker.
            source: News source name.
            published_at: ISO 8601 publication datetime string.
            sentiment: Sentiment label (Positive / Negative / Neutral).
            sentiment_score: Numeric sentiment score 0–1.
            url: Article URL.
            doc_id: Optional explicit ID.

        Returns:
            Document ID.
        """
        text = f"{title}\n\n{content}"
        metadata: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "source": source,
            "published_at": published_at,
            "title": title[:200],  # ChromaDB metadata values must be str/int/float/bool
            "added_at": datetime.utcnow().isoformat(),
        }
        if sentiment:
            metadata["sentiment"] = sentiment
        if sentiment_score is not None:
            metadata["sentiment_score"] = float(sentiment_score)
        if url:
            metadata["url"] = url[:500]

        return self.add_document(text=text, metadata=metadata, doc_id=doc_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB only accepts str, int, float, bool metadata values.
        Convert or drop anything else.
        """
        sanitized: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif v is None:
                pass  # skip None values
            else:
                sanitized[k] = str(v)
        return sanitized
