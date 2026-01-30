"""
Vector Store Manager using ChromaDB
Handles storage and retrieval of document embeddings
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manage vector storage and retrieval with ChromaDB"""
    
    def __init__(self, persist_directory: str = "/home/khaldoun/prjt_vap/chroma_db",
                 collection_name: str = "iso_questions"):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = None
        
        logger.info(f"Vector store initialized at {persist_directory}")
    
    def create_collection(self, embedding_function=None, reset: bool = False):
        """
        Create or get a collection
        
        Args:
            embedding_function: Optional embedding function for ChromaDB
            reset: If True, delete existing collection and create new one
        """
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                logger.debug(f"No existing collection to delete: {e}")
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Collection '{self.collection_name}' ready")
        return self.collection
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Add documents with embeddings to the vector store
        
        Args:
            documents: List of documents with 'id', 'content', 'embedding', and 'metadata'
            batch_size: Batch size for adding documents
        """
        if self.collection is None:
            self.create_collection()
        
        total_docs = len(documents)
        logger.info(f"Adding {total_docs} documents to vector store")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            
            ids = [doc['id'] for doc in batch]
            contents = [doc['content'] for doc in batch]
            embeddings = [doc['embedding'].tolist() for doc in batch]
            
            # Prepare metadata - ChromaDB requires all values to be strings, ints, floats, or bools
            metadatas = []
            for doc in batch:
                metadata = doc.get('metadata', {})
                # Convert all metadata values to strings for ChromaDB compatibility
                clean_metadata = {
                    str(k): str(v) for k, v in metadata.items()
                }
                metadatas.append(clean_metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
            
            logger.info(f"Added batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}")
        
        logger.info(f"Successfully added {total_docs} documents")
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5,
              filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"iso_standard": "iso_27001"})
            
        Returns:
            Dictionary with search results
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        # Convert numpy array to list
        query_embedding_list = query_embedding.tolist()
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results,
            where=filter_metadata  # Optional metadata filtering
        )
        
        return results
    
    def search_by_text(self, query_text: str, n_results: int = 5,
                      filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Search using text query (requires embedding in the caller)
        This is a placeholder - actual embedding should be done before calling
        """
        raise NotImplementedError("Use Retriever class for text-based search")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        if self.collection is None:
            return {"error": "Collection not initialized"}
        
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory)
        }
    
    def delete_collection(self):
        """Delete the current collection"""
        if self.collection is not None:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info(f"Deleted collection: {self.collection_name}")
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        collections = self.client.list_collections()
        return [col.name for col in collections]


class FAISSVectorStore:
    """Alternative vector store using FAISS (faster for large datasets)"""
    
    def __init__(self, persist_directory: str = "/home/khaldoun/prjt_vap/faiss_db"):
        """
        Initialize FAISS vector store
        
        Args:
            persist_directory: Directory to persist the index
        """
        import faiss
        import pickle
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.documents = []
        self.faiss = faiss
        
        logger.info(f"FAISS vector store initialized")
    
    def create_index(self, dimension: int):
        """Create a FAISS index"""
        # Using L2 distance (can be changed to IP for cosine with normalized vectors)
        self.index = self.faiss.IndexFlatL2(dimension)
        logger.info(f"Created FAISS index with dimension {dimension}")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to FAISS index"""
        if self.index is None:
            # Get dimension from first document
            dimension = len(documents[0]['embedding'])
            self.create_index(dimension)
        
        # Normalize embeddings for cosine similarity
        embeddings = np.array([doc['embedding'] for doc in documents])
        
        # Normalize
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents for retrieval
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        # Normalize query
        query_norm = query_embedding.reshape(1, -1).copy()
        self.faiss.normalize_L2(query_norm)
        
        # Search
        distances, indices = self.index.search(query_norm, k)
        
        # Retrieve documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['distance'] = float(distance)
                result['similarity'] = 1 / (1 + float(distance))  # Convert distance to similarity
                results.append(result)
        
        return results
    
    def save(self, filename: str = "index.faiss"):
        """Save index and documents to disk"""
        import pickle
        
        index_path = self.persist_directory / filename
        docs_path = self.persist_directory / "documents.pkl"
        
        # Save index
        self.faiss.write_index(self.index, str(index_path))
        
        # Save documents
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved FAISS index and documents")
    
    def load(self, filename: str = "index.faiss"):
        """Load index and documents from disk"""
        import pickle
        
        index_path = self.persist_directory / filename
        docs_path = self.persist_directory / "documents.pkl"
        
        # Load index
        self.index = self.faiss.read_index(str(index_path))
        
        # Load documents
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        logger.info(f"Loaded FAISS index with {len(self.documents)} documents")


if __name__ == "__main__":
    # Test the vector store
    print("\n=== Testing ChromaDB Vector Store ===")
    
    # Create dummy documents with embeddings
    test_docs = [
        {
            'id': 'test_001',
            'content': 'Have you identified internal issues?',
            'embedding': np.random.rand(384),  # Dummy embedding
            'metadata': {'iso_standard': 'iso_27001', 'title': 'Clause 4'}
        },
        {
            'id': 'test_002',
            'content': 'Is there a security policy?',
            'embedding': np.random.rand(384),
            'metadata': {'iso_standard': 'iso_27001', 'title': 'Clause 5'}
        }
    ]
    
    # Create vector store
    store = VectorStore(persist_directory="/tmp/test_chroma", collection_name="test")
    store.create_collection(reset=True)
    
    # Add documents
    store.add_documents(test_docs)
    
    # Get stats
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Search
    query_emb = np.random.rand(384)
    results = store.search(query_emb, n_results=2)
    print(f"\nSearch results: {len(results['ids'][0])} documents found")
