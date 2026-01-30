"""
Semantic Retriever
Combines embeddings and vector store for semantic search
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

from .embeddings import EmbeddingGenerator, OpenAIEmbeddingGenerator
from .vector_store import VectorStore, FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Retriever for semantic search over ISO documents"""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_generator: EmbeddingGenerator,
                 use_reranking: bool = False):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            use_reranking: Whether to use reranking for better results
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.use_reranking = use_reranking
        
        logger.info("Semantic retriever initialized")
    
    def retrieve(self, 
                query: str, 
                top_k: int = 5,
                filter_standard: Optional[str] = None,
                filter_clause: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query text
            top_k: Number of documents to retrieve
            filter_standard: Optional filter by ISO standard (e.g., "iso_27001")
            filter_clause: Optional filter by clause (e.g., "Clause 4")
            
        Returns:
            List of relevant documents with scores
        """
        logger.info(f"Retrieving documents for query: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Prepare metadata filter
        metadata_filter = {}
        if filter_standard:
            metadata_filter['iso_standard'] = filter_standard
        if filter_clause:
            metadata_filter['title'] = filter_clause
        
        # Search in vector store
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k * 2 if self.use_reranking else top_k,  # Get more if reranking
            filter_metadata=metadata_filter if metadata_filter else None
        )
        
        # Format results
        documents = self._format_results(search_results, top_k)
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
    
    def _format_results(self, search_results: Dict, top_k: int) -> List[Dict]:
        """
        Format ChromaDB results into a clean structure
        
        Args:
            search_results: Raw search results from ChromaDB
            top_k: Number of results to return
            
        Returns:
            List of formatted documents
        """
        documents = []
        
        # ChromaDB returns results in a specific format
        ids = search_results.get('ids', [[]])[0]
        docs = search_results.get('documents', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]
        
        for doc_id, content, metadata, distance in zip(ids, docs, metadatas, distances):
            # Convert distance to similarity score (closer to 1 is better)
            similarity = 1 / (1 + distance)
            
            doc = {
                'id': doc_id,
                'content': content,
                'metadata': metadata,
                'score': similarity,
                'distance': distance
            }
            documents.append(doc)
        
        # If using reranking, apply it here
        if self.use_reranking and len(documents) > top_k:
            documents = self._rerank(documents)[:top_k]
        
        return documents[:top_k]
    
    def _rerank(self, documents: List[Dict]) -> List[Dict]:
        """
        Rerank documents for better relevance (placeholder for future implementation)
        
        Args:
            documents: List of documents to rerank
            
        Returns:
            Reranked documents
        """
        # Simple reranking based on score
        # In production, you could use cross-encoders or other reranking methods
        return sorted(documents, key=lambda x: x['score'], reverse=True)
    
    def retrieve_by_labels(self, 
                          labels: List[str], 
                          top_k: int = 5,
                          filter_standard: Optional[str] = None) -> List[Dict]:
        """
        Retrieve documents that match specific labels
        
        Args:
            labels: List of labels to search for
            top_k: Number of documents to retrieve
            filter_standard: Optional filter by ISO standard
            
        Returns:
            List of relevant documents
        """
        # Create a query from labels
        query = " ".join(labels)
        logger.info(f"Retrieving documents for labels: {labels}")
        
        return self.retrieve(query, top_k=top_k, filter_standard=filter_standard)
    
    def retrieve_hybrid(self,
                       query: str,
                       labels: Optional[List[str]] = None,
                       top_k: int = 5,
                       filter_standard: Optional[str] = None) -> List[Dict]:
        """
        Hybrid retrieval combining query text and labels
        
        Args:
            query: User query text
            labels: Optional list of detected labels
            top_k: Number of documents to retrieve
            filter_standard: Optional filter by ISO standard
            
        Returns:
            List of relevant documents
        """
        if labels:
            # Enhance query with labels
            enhanced_query = f"{query} {' '.join(labels)}"
        else:
            enhanced_query = query
        
        logger.info(f"Hybrid retrieval with enhanced query: '{enhanced_query}'")
        return self.retrieve(enhanced_query, top_k=top_k, filter_standard=filter_standard)
    
    def get_similar_questions(self, question_id: str, top_k: int = 5) -> List[Dict]:
        """
        Find similar questions to a given question
        
        Args:
            question_id: ID of the reference question
            top_k: Number of similar questions to retrieve
            
        Returns:
            List of similar questions
        """
        # This would require storing the question in a way we can retrieve it
        # For now, this is a placeholder
        logger.warning("get_similar_questions not fully implemented")
        return []


class MultiIndexRetriever:
    """Retriever that searches across multiple indices/collections"""
    
    def __init__(self, retrievers: Dict[str, SemanticRetriever]):
        """
        Initialize multi-index retriever
        
        Args:
            retrievers: Dictionary mapping index names to retriever instances
        """
        self.retrievers = retrievers
        logger.info(f"Multi-index retriever initialized with {len(retrievers)} indices")
    
    def retrieve(self, query: str, indices: Optional[List[str]] = None, 
                top_k_per_index: int = 3) -> Dict[str, List[Dict]]:
        """
        Retrieve from multiple indices
        
        Args:
            query: User query
            indices: List of index names to search (None = all)
            top_k_per_index: Number of results per index
            
        Returns:
            Dictionary mapping index names to results
        """
        if indices is None:
            indices = list(self.retrievers.keys())
        
        results = {}
        for index_name in indices:
            if index_name in self.retrievers:
                retriever = self.retrievers[index_name]
                results[index_name] = retriever.retrieve(query, top_k=top_k_per_index)
        
        return results
    
    def retrieve_merged(self, query: str, indices: Optional[List[str]] = None,
                       top_k: int = 5) -> List[Dict]:
        """
        Retrieve from multiple indices and merge results
        
        Args:
            query: User query
            indices: List of index names to search
            top_k: Total number of results to return
            
        Returns:
            Merged and sorted list of results
        """
        all_results = self.retrieve(query, indices, top_k_per_index=top_k)
        
        # Merge all results
        merged = []
        for index_name, docs in all_results.items():
            for doc in docs:
                doc['source_index'] = index_name
                merged.append(doc)
        
        # Sort by score and return top_k
        merged.sort(key=lambda x: x['score'], reverse=True)
        return merged[:top_k]


if __name__ == "__main__":
    # Test retriever (requires initialized vector store)
    print("\n=== Testing Semantic Retriever ===")
    print("Note: This requires an initialized vector store with documents")
    print("Run the full pipeline to test retrieval")
