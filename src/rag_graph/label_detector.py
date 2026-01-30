"""
Label Detector
Detects relevant labels/keywords from user queries using NLP
"""

from typing import List, Dict, Set, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelDetector:
    """Detect labels and keywords from user queries"""
    
    def __init__(self):
        """Initialize the label detector"""
        # Common ISO security keywords mapping
        self.keyword_mapping = {
            # Backup & Recovery
            'backup': ['backup', 'sauvegarde', 'save', 'copy'],
            'restore': ['restore', 'restauration', 'recovery', 'récupération'],
            'disaster recovery': ['disaster', 'sinistre', 'catastrophe', 'disaster recovery'],
            
            # Risk Management
            'risk': ['risk', 'risque', 'threat', 'menace'],
            'vulnerability': ['vulnerability', 'vulnérabilité', 'weakness', 'faiblesse'],
            'risk assessment': ['risk assessment', 'évaluation des risques', 'risk analysis'],
            
            # Security Policy
            'policy': ['policy', 'politique', 'procedure', 'procédure'],
            'security policy': ['security policy', 'politique de sécurité'],
            'documentation': ['documentation', 'document', 'documented', 'documenté'],
            
            # Access Control
            'access control': ['access control', "contrôle d'accès", 'authentication'],
            'authorization': ['authorization', 'autorisation', 'permission'],
            'identity': ['identity', 'identité', 'user', 'utilisateur'],
            
            # ISMS
            'isms': ['isms', 'smsi', 'information security management'],
            'isms scope': ['scope', 'périmètre', 'boundary', 'frontière'],
            'isms objectives': ['objectives', 'objectifs', 'goals', 'buts'],
            
            # Monitoring
            'monitoring': ['monitoring', 'surveillance', 'supervision', 'watch'],
            'audit': ['audit', 'review', 'révision', 'inspection'],
            'incident': ['incident', 'event', 'événement', 'breach', 'violation'],
            
            # Personnel
            'employee': ['employee', 'employé', 'staff', 'personnel'],
            'training': ['training', 'formation', 'awareness', 'sensibilisation'],
            'competence': ['competence', 'compétence', 'skill', 'compétence'],
            
            # Data Protection
            'data': ['data', 'données', 'information'],
            'privacy': ['privacy', 'confidentialité', 'vie privée'],
            'encryption': ['encryption', 'chiffrement', 'cryptography', 'cryptographie'],
            
            # Cloud
            'cloud': ['cloud', 'nuage', 'cloud computing'],
            'service provider': ['service provider', 'fournisseur de services', 'provider'],
            
            # General
            'management': ['management', 'gestion', 'governance', 'gouvernance'],
            'compliance': ['compliance', 'conformité', 'adherence'],
            'testing': ['testing', 'test', 'validation', 'vérification']
        }
        
        # ISO standard names
        self.standards = {
            'iso 27001': ['27001', 'iso 27001', 'iso27001'],
            'iso 27002': ['27002', 'iso 27002', 'iso27002'],
            'iso 27017': ['27017', 'iso 27017', 'iso27017'],
            'iso 27018': ['27018', 'iso 27018', 'iso27018'],
            'iso 27701': ['27701', 'iso 27701', 'iso27701']
        }
        
        # Clause patterns
        self.clause_patterns = [
            r'clause\s+(\d+)',
            r'article\s+(\d+)',
            r'section\s+(\d+)',
            r'point\s+(\d+)'
        ]
        
        logger.info("OK - Label detector initialized")
    
    def detect_labels(self, query: str) -> List[str]:
        """
        Detect relevant labels from a query
        
        Args:
            query: User query text
            
        Returns:
            List of detected labels
        """
        query_lower = query.lower()
        detected = set()
        
        # Check each keyword category
        for label, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected.add(label)
                    break
        
        logger.info(f"Detected labels: {detected}")
        return list(detected)
    
    def detect_standard(self, query: str) -> Optional[str]:
        """
        Detect ISO standard from query
        
        Args:
            query: User query text
            
        Returns:
            ISO standard name or None
        """
        query_lower = query.lower()
        
        for standard, patterns in self.standards.items():
            for pattern in patterns:
                if pattern in query_lower:
                    logger.info(f"Detected standard: {standard}")
                    return f"iso_{standard.split()[-1]}"
        
        return None
    
    def detect_clause(self, query: str) -> Optional[str]:
        """
        Detect clause number from query
        
        Args:
            query: User query text
            
        Returns:
            Clause identifier or None
        """
        query_lower = query.lower()
        
        for pattern in self.clause_patterns:
            match = re.search(pattern, query_lower)
            if match:
                clause_num = match.group(1)
                logger.info(f"Detected clause: {clause_num}")
                return f"Clause {clause_num}"
        
        return None
    
    def extract_context(self, query: str) -> Dict:
        """
        Extract all context from query
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with labels, standard, and clause
        """
        context = {
            'labels': self.detect_labels(query),
            'standard': self.detect_standard(query),
            'clause': self.detect_clause(query),
            'query': query
        }
        
        logger.info(f"Extracted context: {context}")
        return context
    
    def get_all_labels(self) -> List[str]:
        """Get all available labels"""
        return list(self.keyword_mapping.keys())
    
    def get_related_labels(self, label: str) -> List[str]:
        """
        Get labels related to a given label (simplified)
        
        Args:
            label: Base label
            
        Returns:
            List of related labels
        """
        # Simple grouping by category
        categories = {
            'backup_recovery': ['backup', 'restore', 'disaster recovery'],
            'risk': ['risk', 'vulnerability', 'risk assessment'],
            'policy': ['policy', 'security policy', 'documentation'],
            'access': ['access control', 'authorization', 'identity'],
            'isms': ['isms', 'isms scope', 'isms objectives'],
            'monitoring': ['monitoring', 'audit', 'incident'],
            'people': ['employee', 'training', 'competence'],
            'data': ['data', 'privacy', 'encryption']
        }
        
        # Find category containing the label
        for category, labels in categories.items():
            if label in labels:
                return [l for l in labels if l != label]
        
        return []


class AdvancedLabelDetector(LabelDetector):
    """Advanced label detector using sentence embeddings"""
    
    def __init__(self, embedding_model=None):
        """
        Initialize with optional embedding model
        
        Args:
            embedding_model: Optional EmbeddingGenerator instance
        """
        super().__init__()
        self.embedding_model = embedding_model
        
        if embedding_model:
            logger.info("Using semantic similarity for label detection")
    
    def detect_labels_semantic(self, query: str, threshold: float = 0.5) -> List[str]:
        """
        Detect labels using semantic similarity
        
        Args:
            query: User query
            threshold: Similarity threshold
            
        Returns:
            List of detected labels
        """
        if not self.embedding_model:
            return self.detect_labels(query)
        
        # Get query embedding
        query_emb = self.embedding_model.embed_text(query)
        
        # Get embeddings for all label descriptions
        label_texts = list(self.keyword_mapping.keys())
        label_embs = self.embedding_model.embed_batch(label_texts)
        
        # Compute similarities
        detected = []
        for label, label_emb in zip(label_texts, label_embs):
            similarity = self.embedding_model.compute_similarity(query_emb, label_emb)
            if similarity > threshold:
                detected.append(label)
        
        logger.info(f"Semantically detected labels: {detected}")
        return detected


if __name__ == "__main__":
    # Test the detector
    print("\n" + "=" * 80)
    print(" Label Detector Test")
    print("=" * 80)
    
    detector = LabelDetector()
    
    # Test queries
    test_queries = [
        "Questions sur les sauvegardes de données",
        "J'ai besoin d'un questionnaire pour l'évaluation des risques ISO 27001",
        "Politique de sécurité et sensibilisation des employés",
        "Cloud computing et protection de la vie privée ISO 27018",
        "Clause 4 - Contexte de l'organisation"
    ]
    
    for query in test_queries:
        print(f"\n Query: '{query}'")
        context = detector.extract_context(query)
        print(f"   Labels: {context['labels']}")
        print(f"   Standard: {context['standard']}")
        print(f"   Clause: {context['clause']}")
