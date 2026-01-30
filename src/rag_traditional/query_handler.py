"""
Query Handler - Main Interface for RAG System
Orchestrates the complete RAG pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional
import logging
from datetime import datetime

from src.utils.data_loader import ISODataLoader
from src.rag_traditional.embeddings import EmbeddingGenerator
from src.rag_traditional.vector_store import VectorStore
from src.rag_traditional.retriever import SemanticRetriever
from src.llm.llm_interface import LLMFactory, BaseLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ISORAGSystem:
    """Complete RAG system for ISO questionnaire generation"""
    
    def __init__(self,
                 data_method: int = 1,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 persist_directory: str = "/home/khaldoun/prjt_vap/chroma_db",
                 rebuild_index: bool = False):
        """
        Initialize the RAG system
        
        Args:
            data_method: Which method data to use (1 or 2)
            embedding_model: Name of the embedding model
            llm_provider: LLM provider ('openai' or 'anthropic')
            llm_model: LLM model name
            persist_directory: Directory to persist vector store
            rebuild_index: Whether to rebuild the vector index
        """
        logger.info("Initializing ISO RAG System...")
        
        self.data_method = data_method
        
        # Initialize components
        logger.info("Loading data...")
        self.data_loader = ISODataLoader()
        
        logger.info("Initializing embedding model...")
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        logger.info("Initializing vector store...")
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            collection_name=f"iso_questions_method_{data_method}"
        )
        
        logger.info("Initializing LLM...")
        self.llm = LLMFactory.create_llm(
            provider=llm_provider,
            model=llm_model
        )
        
        # Build or load index
        if rebuild_index:
            self._build_index()
        else:
            self.vector_store.create_collection()
            stats = self.vector_store.get_collection_stats()
            if stats['document_count'] == 0:
                logger.info("Vector store is empty. Building index...")
                self._build_index()
        
        # Initialize retriever
        logger.info("Initializing retriever...")
        self.retriever = SemanticRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )
        
        logger.info("OK - ISO RAG System initialized successfully!")
    
    def _build_index(self):
        """Build the vector index from CSV files"""
        logger.info(f"Building index from method {self.data_method} data...")
        
        # Create/reset collection
        self.vector_store.create_collection(reset=True)
        
        # Load documents
        documents = self.data_loader.get_documents_for_rag(method=self.data_method)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        documents = self.embedding_generator.embed_documents(documents)
        
        # Add to vector store
        logger.info("Adding documents to vector store...")
        self.vector_store.add_documents(documents)
        
        logger.info("OK - Index built successfully!")
    
    def query(self, 
             user_query: str,
             top_k: int = 10,
             filter_standard: Optional[str] = None,
             filter_clause: Optional[str] = None,
             return_sources: bool = True) -> Dict:
        """
        Process a user query and generate response
        
        Args:
            user_query: The user's question or request
            top_k: Number of relevant documents to retrieve
            filter_standard: Optional ISO standard filter (e.g., "iso_27001")
            filter_clause: Optional clause filter (e.g., "Clause 4")
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optionally source documents
        """
        logger.info(f"Processing query: '{user_query}'")
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(
            query=user_query,
            top_k=top_k,
            filter_standard=filter_standard,
            filter_clause=filter_clause
        )
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            return {
                'answer': "Désolé, je n'ai pas trouvé de questions pertinentes pour votre requête.",
                'sources': [],
                'query': user_query,
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        
        # Generate response with LLM
        logger.info("Generating response with LLM...")
        response = self.llm.generate_with_context(
            query=user_query,
            context_docs=relevant_docs
        )
        
        result = {
            'answer': response,
            'query': user_query,
            'timestamp': datetime.now().isoformat(),
            'num_sources': len(relevant_docs)
        }
        
        if return_sources:
            result['sources'] = relevant_docs
        
        return result
    
    def generate_questionnaire(self,
                              topic: str,
                              iso_standard: Optional[str] = None,
                              num_questions: int = 15) -> Dict:
        """
        Generate a custom questionnaire for a specific topic
        
        Args:
            topic: Topic or theme for the questionnaire (e.g., "backup de données")
            iso_standard: Optional ISO standard to focus on
            num_questions: Number of questions to include
            
        Returns:
            Dictionary with generated questionnaire
        """
        query = f"Générez un questionnaire d'audit complet sur le thème: {topic}"
        if iso_standard:
            query += f" pour la norme {iso_standard}"
        
        return self.query(
            user_query=query,
            top_k=num_questions,
            filter_standard=iso_standard,
            return_sources=True
        )
    
    def search_questions(self,
                        keywords: List[str],
                        iso_standard: Optional[str] = None,
                        top_k: int = 10) -> List[Dict]:
        """
        Search for questions matching specific keywords
        
        Args:
            keywords: List of keywords to search for
            iso_standard: Optional ISO standard filter
            top_k: Number of questions to return
            
        Returns:
            List of matching questions
        """
        query = " ".join(keywords)
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_standard=iso_standard
        )
    
    def get_statistics(self) -> Dict:
        """
        Get system statistics
        
        Returns:
            Dictionary with statistics
        """
        vector_stats = self.vector_store.get_collection_stats()
        data_stats = self.data_loader.get_statistics(method=self.data_method)
        
        return {
            'vector_store': vector_stats,
            'data': data_stats,
            'embedding_model': self.embedding_generator.model_name,
            'embedding_dimension': self.embedding_generator.get_embedding_dimension()
        }


def main():
    """Example usage of the RAG system"""
    print("\n" + "="*80)
    print(" ISO RAG System - Traditional Approach")
    print("="*80)
    
    # Initialize system
    print("\n[1/3] Initializing RAG system...")
    rag = ISORAGSystem(
        data_method=1,
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="openai",
        rebuild_index=False  # Set to True to rebuild index
    )
    
    # Get statistics
    print("\n[2/3] System Statistics:")
    stats = rag.get_statistics()
    print(f"  • Total documents: {stats['data']['total_questions']}")
    print(f"  • ISO Standards: {', '.join(stats['data']['standards'])}")
    print(f"  • Embedding dimension: {stats['embedding_dimension']}")
    
    # Example queries
    print("\n[3/3] Example Queries:")
    print("-" * 80)
    
    # Query 1: Backup
    print("\n Query 1: 'J'ai besoin d'un questionnaire pour les backup de données ISO 27001'")
    result = rag.generate_questionnaire(
        topic="backup de données",
        iso_standard="iso_27001",
        num_questions=10
    )
    print(f"\n Generated Answer:\n{result['answer']}")
    print(f"\n📚 Based on {result['num_sources']} source documents")
    
    # Query 2: Risk assessment
    print("\n" + "-" * 80)
    print("\n Query 2: 'Questions sur l'évaluation des risques'")
    result2 = rag.query(
        user_query="Questions sur l'évaluation des risques et la gestion des risques",
        top_k=8,
        return_sources=False
    )
    print(f"\n Generated Answer:\n{result2['answer']}")
    
    print("\n" + "="*80)
    print("OK - Demo completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
