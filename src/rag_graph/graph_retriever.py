"""
Graph Retriever
Retrieves questions from Neo4j knowledge graph using Cypher queries
"""

from typing import List, Dict, Optional
import logging
from neo4j import GraphDatabase

from .label_detector import LabelDetector
from .cypher_generator import CypherQueryGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieve questions from Neo4j knowledge graph"""
    
    def __init__(self,
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password"):
        """
        Initialize Neo4j connection and components
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("OK - Connected to Neo4j")
        except Exception as e:
            logger.error(f"ERREUR - Failed to connect to Neo4j: {e}")
            raise
        
        self.label_detector = LabelDetector()
        self.query_generator = CypherQueryGenerator()
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def retrieve(self,
                query: str,
                top_k: int = 10) -> List[Dict]:
        """
        Retrieve relevant questions based on user query
        
        Args:
            query: User query text
            top_k: Maximum number of results
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Retrieving questions for: '{query}'")
        
        # Detect context from query
        context = self.label_detector.extract_context(query)
        
        labels = context['labels']
        standard = context['standard']
        clause = context['clause']
        
        logger.info(f"Context: labels={labels}, standard={standard}, clause={clause}")
        
        # Generate Cypher query
        if not labels and not standard and not clause:
            # Fallback: get random sample
            cypher = f"MATCH (q:Question) RETURN q.id as id, q.text as text LIMIT {top_k}"
        else:
            cypher = self.query_generator.generate_multi_hop_query(
                labels=labels,
                standard=standard,
                limit=top_k
            )
        
        # Execute query
        results = self._execute_query(cypher)
        
        # Format results
        formatted = self._format_results(results)
        
        logger.info(f"Retrieved {len(formatted)} questions")
        return formatted
    
    def retrieve_by_labels(self,
                          labels: List[str],
                          standard: Optional[str] = None,
                          top_k: int = 10) -> List[Dict]:
        """
        Retrieve questions by specific labels
        
        Args:
            labels: List of labels
            standard: Optional ISO standard filter
            top_k: Maximum results
            
        Returns:
            List of questions
        """
        cypher = self.query_generator.generate_label_query(
            labels=labels,
            standard=standard,
            limit=top_k
        )
        
        results = self._execute_query(cypher)
        return self._format_results(results)
    
    def retrieve_by_clause(self,
                          clause: str,
                          standard: Optional[str] = None,
                          top_k: int = 20) -> List[Dict]:
        """
        Retrieve all questions for a clause
        
        Args:
            clause: Clause title
            standard: Optional standard filter
            top_k: Maximum results
            
        Returns:
            List of questions
        """
        cypher = self.query_generator.generate_clause_questions_query(
            clause=clause,
            standard=standard,
            limit=top_k
        )
        
        results = self._execute_query(cypher)
        return self._format_results(results)
    
    def get_related_questions(self,
                             question_id: str,
                             top_k: int = 5) -> List[Dict]:
        """
        Find questions related to a given question
        
        Args:
            question_id: Question ID
            top_k: Maximum results
            
        Returns:
            List of related questions
        """
        cypher = self.query_generator.generate_related_questions_query(
            question_id=question_id,
            limit=top_k
        )
        
        results = self._execute_query(cypher)
        return self._format_results(results)
    
    def get_standard_overview(self, standard: str) -> Dict:
        """
        Get overview statistics for a standard
        
        Args:
            standard: ISO standard name
            
        Returns:
            Dictionary with statistics
        """
        cypher = self.query_generator.generate_standard_overview_query(standard)
        
        with self.driver.session() as session:
            result = session.run(cypher)
            record = result.single()
            
            if record:
                return {
                    'standard': record['standard'],
                    'total_questions': record['total_questions'],
                    'clauses': record['clauses'],
                    'labels': record['all_labels']
                }
        
        return {}
    
    def search_by_text(self,
                      text: str,
                      top_k: int = 10) -> List[Dict]:
        """
        Simple text search in question content
        
        Args:
            text: Search text
            top_k: Maximum results
            
        Returns:
            List of questions
        """
        cypher = f"""
        MATCH (q:Question)
        WHERE toLower(q.text) CONTAINS toLower('{text}')
        MATCH (q)-[:BELONGS_TO_STANDARD]->(s:Standard)
        MATCH (q)-[:BELONGS_TO_CLAUSE]->(c:Clause)
        MATCH (q)-[:HAS_LABEL]->(l:Label)
        WITH q, s, c, collect(DISTINCT l.name) as labels
        RETURN q.id as id,
               q.text as text,
               s.name as standard,
               c.title as clause,
               labels
        LIMIT {top_k}
        """
        
        results = self._execute_query(cypher)
        return self._format_results(results)
    
    def _execute_query(self, cypher: str) -> List[Dict]:
        """Execute Cypher query and return results"""
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            logger.error(f"Query was: {cypher}")
            return []
    
    def _format_results(self, results: List[Dict]) -> List[Dict]:
        """
        Format results in consistent structure
        
        Args:
            results: Raw query results
            
        Returns:
            Formatted results
        """
        formatted = []
        
        for result in results:
            formatted_result = {
                'id': result.get('id', 'unknown'),
                'content': result.get('text', ''),
                'metadata': {
                    'iso_standard': result.get('standard', 'unknown'),
                    'title': result.get('clause', 'unknown'),
                    'labels': ', '.join(result.get('labels', [])) if isinstance(result.get('labels'), list) else result.get('labels', '')
                },
                'score': 1.0  # Graph queries don't have similarity scores
            }
            
            # Add relevance if available
            if 'relevance' in result:
                formatted_result['score'] = result['relevance'] / 10.0
            elif 'direct_match_count' in result:
                formatted_result['score'] = result['direct_match_count'] / 5.0
            
            formatted.append(formatted_result)
        
        return formatted
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        stats_query = """
        MATCH (q:Question) WITH count(q) as questions
        MATCH (l:Label) WITH questions, count(l) as labels
        MATCH (s:Standard) WITH questions, labels, count(s) as standards
        MATCH (c:Clause) WITH questions, labels, standards, count(c) as clauses
        MATCH ()-[r]->() WITH questions, labels, standards, clauses, count(r) as relationships
        RETURN questions, labels, standards, clauses, relationships
        """
        
        with self.driver.session() as session:
            result = session.run(stats_query)
            record = result.single()
            
            if record:
                return dict(record)
        
        return {}


class HybridRetriever:
    """Combines graph retrieval with vector-based retrieval"""
    
    def __init__(self,
                 graph_retriever: GraphRetriever,
                 vector_retriever=None):
        """
        Initialize hybrid retriever
        
        Args:
            graph_retriever: GraphRetriever instance
            vector_retriever: Optional SemanticRetriever instance
        """
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever
        
        logger.info("OK - Hybrid retriever initialized")
    
    def retrieve(self,
                query: str,
                top_k: int = 10,
                graph_weight: float = 0.5) -> List[Dict]:
        """
        Retrieve using both graph and vector methods
        
        Args:
            query: User query
            top_k: Total number of results
            graph_weight: Weight for graph results (0-1)
            
        Returns:
            Combined results
        """
        results = []
        
        # Graph retrieval
        graph_results = self.graph_retriever.retrieve(query, top_k=top_k)
        for result in graph_results:
            result['score'] = result['score'] * graph_weight
            result['source'] = 'graph'
        results.extend(graph_results)
        
        # Vector retrieval if available
        if self.vector_retriever:
            vector_results = self.vector_retriever.retrieve(query, top_k=top_k)
            for result in vector_results:
                result['score'] = result['score'] * (1 - graph_weight)
                result['source'] = 'vector'
            results.extend(vector_results)
        
        # Remove duplicates and sort by score
        seen_ids = set()
        unique_results = []
        for result in sorted(results, key=lambda x: x['score'], reverse=True):
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        return unique_results[:top_k]


if __name__ == "__main__":
    # Test the retriever
    print("\n" + "=" * 80)
    print(" Graph Retriever Test")
    print("=" * 80)
    
    try:
        retriever = GraphRetriever(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        
        # Test query
        query = "Questions sur les sauvegardes et la restauration"
        print(f"\n Query: '{query}'")
        
        results = retriever.retrieve(query, top_k=5)
        
        print(f"\nOK - Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['content'][:80]}...")
            print(f"    Standard: {result['metadata']['iso_standard']}")
            print(f"    Score: {result['score']:.2f}")
        
        retriever.close()
        
    except Exception as e:
        print(f"\nERREUR - Error: {e}")
        print("\nMake sure Neo4j is running and the graph is built.")
