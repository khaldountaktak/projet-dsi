"""
LLM Interface for RAG System
Handles interaction with various LLM providers (OpenAI, Anthropic, etc.)
"""

from typing import List, Dict, Optional, Any
import logging
import os
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Base class for LLM implementations"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def generate_with_context(self, query: str, context_docs: List[Dict], **kwargs) -> str:
        """Generate text with retrieved context"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.3):
        """
        Initialize OpenAI LLM
        
        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: Model name (gpt-4, gpt-4o-mini, gpt-3.5-turbo, etc.)
            temperature: Temperature for generation (0-2)
        """
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        
        logger.info(f"Initialized OpenAI LLM with model: {model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        response = self.client.chat.completions.create(
            model=kwargs.get('model', self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', 2000)
        )
        
        return response.choices[0].message.content
    
    def generate_with_context(self, query: str, context_docs: List[Dict], **kwargs) -> str:
        """
        Generate answer using retrieved context
        
        Args:
            query: User query
            context_docs: List of retrieved documents
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response
        """
        # Format context
        context = self._format_context(context_docs)
        
        # Create prompt
        prompt = self._create_rag_prompt(query, context, kwargs.get('system_prompt'))
        
        # Generate
        response = self.client.chat.completions.create(
            model=kwargs.get('model', self.model),
            messages=[
                {"role": "system", "content": kwargs.get('system_prompt', self._get_default_system_prompt())},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', 2000)
        )
        
        return response.choices[0].message.content
    
    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.get('metadata', {})
            iso_standard = metadata.get('iso_standard', 'N/A')
            title = metadata.get('title', 'N/A')
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Standard: {iso_standard}\n"
                f"Clause: {title}\n"
                f"Question: {doc['content']}\n"
                f"Labels: {metadata.get('labels', 'N/A')}\n"
            )
        
        return "\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG"""
        return """Vous êtes un expert en normes ISO de sécurité de l'information (ISO 27001, 27002, 27017, 27018, 27701).
Votre rôle est d'aider les utilisateurs à créer des questionnaires d'audit personnalisés basés sur leurs besoins.

Utilisez les documents fournis comme contexte pour générer des questionnaires pertinents et bien structurés.
Chaque question doit être claire, spécifique et conforme aux normes ISO.

Si l'utilisateur demande un questionnaire sur un thème spécifique, utilisez les questions similaires dans le contexte
pour créer un questionnaire cohérent et complet."""
    
    def _create_rag_prompt(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Create RAG prompt combining query and context"""
        return f"""Contexte (questions ISO pertinentes):
{context}

Requête utilisateur: {query}

Générez un questionnaire structuré et personnalisé basé sur la requête de l'utilisateur et le contexte fourni.
Organisez les questions de manière logique et ajoutez des titres de section si nécessaire."""


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "claude-3-5-sonnet-20241022",
                 temperature: float = 0.3):
        """
        Initialize Anthropic LLM
        
        Args:
            api_key: Anthropic API key (if None, uses environment variable)
            model: Model name (claude-3-opus, claude-3-sonnet, etc.)
            temperature: Temperature for generation
        """
        from anthropic import Anthropic
        
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        
        logger.info(f"Initialized Anthropic LLM with model: {model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        response = self.client.messages.create(
            model=kwargs.get('model', self.model),
            max_tokens=kwargs.get('max_tokens', 2000),
            temperature=kwargs.get('temperature', self.temperature),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_with_context(self, query: str, context_docs: List[Dict], **kwargs) -> str:
        """Generate answer using retrieved context"""
        # Format context
        context = self._format_context(context_docs)
        
        # Create prompt
        system_prompt = kwargs.get('system_prompt', self._get_default_system_prompt())
        user_prompt = self._create_rag_prompt(query, context)
        
        # Generate
        response = self.client.messages.create(
            model=kwargs.get('model', self.model),
            max_tokens=kwargs.get('max_tokens', 2000),
            temperature=kwargs.get('temperature', self.temperature),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        return response.content[0].text
    
    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.get('metadata', {})
            iso_standard = metadata.get('iso_standard', 'N/A')
            title = metadata.get('title', 'N/A')
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Standard: {iso_standard}\n"
                f"Clause: {title}\n"
                f"Question: {doc['content']}\n"
                f"Labels: {metadata.get('labels', 'N/A')}\n"
            )
        
        return "\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG"""
        return """Vous êtes un expert en normes ISO de sécurité de l'information (ISO 27001, 27002, 27017, 27018, 27701).
Votre rôle est d'aider les utilisateurs à créer des questionnaires d'audit personnalisés basés sur leurs besoins.

Utilisez les documents fournis comme contexte pour générer des questionnaires pertinents et bien structurés.
Chaque question doit être claire, spécifique et conforme aux normes ISO.

Si l'utilisateur demande un questionnaire sur un thème spécifique, utilisez les questions similaires dans le contexte
pour créer un questionnaire cohérent et complet."""
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create RAG prompt combining query and context"""
        return f"""Contexte (questions ISO pertinentes):
{context}

Requête utilisateur: {query}

Générez un questionnaire structuré et personnalisé basé sur la requête de l'utilisateur et le contexte fourni.
Organisez les questions de manière logique et ajoutez des titres de section si nécessaire."""


class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation (FREE!)"""
    
    def __init__(self,
                 model: str = "llama3.2",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.3):
        """
        Initialize Ollama LLM (local, free, no API key needed!)
        
        Args:
            model: Model name (llama3.2, mistral, phi3, etc.)
            base_url: Ollama server URL
            temperature: Temperature for generation
        """
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self.model = model
            self.temperature = temperature
            
            logger.info(f"Initialized Ollama LLM with model: {model}")
        except ImportError:
            logger.error("Ollama package not installed. Install with: pip install ollama")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        response = self.client.generate(
            model=kwargs.get('model', self.model),
            prompt=prompt,
            options={
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', 2000)
            }
        )
        return response['response']
    
    def generate_with_context(self, query: str, context_docs: List[Dict], **kwargs) -> str:
        """Generate answer using retrieved context"""
        # Format context
        context = self._format_context(context_docs)
        
        # Create prompt with system instructions
        full_prompt = f"""{self._get_default_system_prompt()}

{self._create_rag_prompt(query, context)}"""
        
        # Generate
        response = self.client.generate(
            model=kwargs.get('model', self.model),
            prompt=full_prompt,
            options={
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', 2000)
            }
        )
        
        return response['response']
    
    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.get('metadata', {})
            iso_standard = metadata.get('iso_standard', 'N/A')
            title = metadata.get('title', 'N/A')
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Standard: {iso_standard}\n"
                f"Clause: {title}\n"
                f"Question: {doc['content']}\n"
                f"Labels: {metadata.get('labels', 'N/A')}\n"
            )
        
        return "\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG"""
        return """Vous êtes un expert en normes ISO de sécurité de l'information (ISO 27001, 27002, 27017, 27018, 27701).
Votre rôle est d'aider les utilisateurs à créer des questionnaires d'audit personnalisés basés sur leurs besoins.

Utilisez les documents fournis comme contexte pour générer des questionnaires pertinents et bien structurés.
Chaque question doit être claire, spécifique et conforme aux normes ISO."""
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create RAG prompt combining query and context"""
        return f"""Contexte (questions ISO pertinentes):
{context}

Requête utilisateur: {query}

Générez un questionnaire structuré basé sur la requête de l'utilisateur et le contexte fourni."""


class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create_llm(provider: str = "openai", **kwargs) -> BaseLLM:
        """
        Create an LLM instance
        
        Args:
            provider: LLM provider ('openai', 'anthropic', or 'ollama')
            **kwargs: Arguments to pass to the LLM constructor
            
        Returns:
            LLM instance
        """
        if provider.lower() == "openai":
            return OpenAILLM(**kwargs)
        elif provider.lower() == "anthropic":
            return AnthropicLLM(**kwargs)
        elif provider.lower() == "ollama":
            return OllamaLLM(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    print("\n=== Testing LLM Interface ===")
    print("Note: Requires API keys to be set in environment variables")
    print("OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    # Example usage
    try:
        llm = LLMFactory.create_llm("openai", model="gpt-4o-mini")
        
        test_docs = [
            {
                'content': 'Are backup procedures documented and tested?',
                'metadata': {
                    'iso_standard': 'iso_27001',
                    'title': 'Clause 8',
                    'labels': 'backup, documentation'
                }
            }
        ]
        
        response = llm.generate_with_context(
            "J'ai besoin de questions sur les backups",
            test_docs
        )
        print(f"\nLLM Response:\n{response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
