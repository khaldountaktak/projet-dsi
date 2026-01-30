"""
Embedding Generator for ISO documents
Creates vector embeddings for semantic search
"""

from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for documents using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model
                       Options:
                       - 'all-MiniLM-L6-v2' (default, fast and efficient)
                       - 'all-mpnet-base-v2' (better quality, slower)
                       - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Model loaded successfully")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Numpy array with embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, 
                   show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Embed a list of documents and add embeddings to them
        
        Args:
            documents: List of document dictionaries with 'content' key
            
        Returns:
            List of documents with added 'embedding' key
        """
        # Extract texts
        texts = [doc['content'] for doc in documents]
        
        # Also include labels and title in the embedding for better retrieval
        enhanced_texts = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            labels = metadata.get('labels', '')
            title = metadata.get('title', '')
            
            # Combine content with metadata for richer embeddings
            enhanced_text = f"{doc['content']} | {title} | {labels}"
            enhanced_texts.append(enhanced_text)
        
        # Generate embeddings
        embeddings = self.embed_batch(enhanced_texts)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        
        logger.info(f"Added embeddings to {len(documents)} documents")
        return documents
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0 to 1)
        """
        from numpy.linalg import norm
        
        similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
        return float(similarity)


class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding generator
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: OpenAI embedding model
        """
        from openai import OpenAI
        import os
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        logger.info(f"Initialized OpenAI embeddings with model: {model}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return np.array(all_embeddings)
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """Embed documents with OpenAI"""
        texts = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            labels = metadata.get('labels', '')
            title = metadata.get('title', '')
            enhanced_text = f"{doc['content']} | {title} | {labels}"
            texts.append(enhanced_text)
        
        embeddings = self.embed_batch(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        
        return documents


if __name__ == "__main__":
    # Test the embedding generator
    print("\n=== Testing Sentence Transformer Embeddings ===")
    generator = EmbeddingGenerator()
    
    test_texts = [
        "Have you identified internal issues that affect your ISMS?",
        "Is there a documented information security policy?",
        "Are backup procedures documented and tested?"
    ]
    
    embeddings = generator.embed_batch(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {generator.get_embedding_dimension()}")
    
    # Test similarity
    sim = generator.compute_similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity between first two texts: {sim:.4f}")
