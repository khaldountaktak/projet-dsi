"""
Cypher Query Generator
Generates Neo4j Cypher queries based on detected labels and context
"""

from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CypherQueryGenerator:
    """Generate Cypher queries for Neo4j based on user context"""
    
    def __init__(self):
        """Initialize the query generator"""
        logger.info("OK - Cypher query generator initialized")
    
    def generate_label_query(self, 
                            labels: List[str], 
                            standard: Optional[str] = None,
                            clause: Optional[str] = None,
                            limit: int = 10) -> str:
        """
        Generate query to find questions by labels
        
        Args:
            labels: List of labels to search for
            standard: Optional ISO standard filter
            clause: Optional clause filter
            limit: Maximum number of results
            
        Returns:
            Cypher query string
        """
        # Build WHERE conditions
        conditions = []
        
        if labels:
            # Match any of the labels
            label_conditions = " OR ".join([f"l.name = '{label}'" for label in labels])
            conditions.append(f"({label_conditions})")
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        # Add standard filter
        standard_filter = ""
        if standard:
            standard_filter = f"AND s.name = '{standard}'"
        
        # Add clause filter
        clause_filter = ""
        if clause:
            clause_filter = f"AND c.title CONTAINS '{clause}'"
        
        query = f"""
        MATCH (q:Question)-[:HAS_LABEL]->(l:Label)
        WHERE {where_clause}
        WITH q, collect(DISTINCT l.name) as labels
        MATCH (q)-[:BELONGS_TO_STANDARD]->(s:Standard)
        {standard_filter}
        MATCH (q)-[:BELONGS_TO_CLAUSE]->(c:Clause)
        {clause_filter}
        RETURN q.id as id, 
               q.text as text, 
               s.name as standard,
               c.title as clause,
               labels
        ORDER BY size(labels) DESC
        LIMIT {limit}
        """
        
        return query.strip()
    
    def generate_related_questions_query(self,
                                        question_id: str,
                                        max_distance: int = 2,
                                        limit: int = 10) -> str:
        """
        Find questions related to a given question
        
        Args:
            question_id: ID of the source question
            max_distance: Maximum relationship distance
            limit: Maximum number of results
            
        Returns:
            Cypher query string
        """
        query = f"""
        MATCH (q1:Question {{id: '{question_id}'}})-[:HAS_LABEL]->(l:Label)<-[:HAS_LABEL]-(q2:Question)
        WHERE q1 <> q2
        WITH q2, collect(DISTINCT l.name) as shared_labels
        MATCH (q2)-[:BELONGS_TO_STANDARD]->(s:Standard)
        MATCH (q2)-[:BELONGS_TO_CLAUSE]->(c:Clause)
        RETURN q2.id as id,
               q2.text as text,
               s.name as standard,
               c.title as clause,
               shared_labels,
               size(shared_labels) as relevance
        ORDER BY relevance DESC
        LIMIT {limit}
        """
        
        return query.strip()
    
    def generate_standard_overview_query(self, standard: str) -> str:
        """
        Get overview of a standard
        
        Args:
            standard: ISO standard name
            
        Returns:
            Cypher query string
        """
        query = f"""
        MATCH (s:Standard {{name: '{standard}'}})
        MATCH (q:Question)-[:BELONGS_TO_STANDARD]->(s)
        WITH s, count(q) as total_questions
        MATCH (c:Clause)-[:PART_OF_STANDARD]->(s)
        WITH s, total_questions, collect(DISTINCT c.title) as clauses
        MATCH (q:Question)-[:BELONGS_TO_STANDARD]->(s)
        MATCH (q)-[:HAS_LABEL]->(l:Label)
        WITH s, total_questions, clauses, collect(DISTINCT l.name) as all_labels
        RETURN s.name as standard,
               total_questions,
               clauses,
               all_labels
        """
        
        return query.strip()
    
    def generate_clause_questions_query(self,
                                       clause: str,
                                       standard: Optional[str] = None,
                                       limit: int = 20) -> str:
        """
        Get all questions for a specific clause
        
        Args:
            clause: Clause title
            standard: Optional standard filter
            limit: Maximum results
            
        Returns:
            Cypher query string
        """
        standard_filter = f"AND c.standard = '{standard}'" if standard else ""
        
        query = f"""
        MATCH (c:Clause)
        WHERE c.title CONTAINS '{clause}' {standard_filter}
        MATCH (q:Question)-[:BELONGS_TO_CLAUSE]->(c)
        MATCH (q)-[:HAS_LABEL]->(l:Label)
        WITH q, c, collect(DISTINCT l.name) as labels
        MATCH (q)-[:BELONGS_TO_STANDARD]->(s:Standard)
        RETURN q.id as id,
               q.text as text,
               s.name as standard,
               c.title as clause,
               labels
        LIMIT {limit}
        """
        
        return query.strip()
    
    def generate_label_co_occurrence_query(self, label: str, limit: int = 10) -> str:
        """
        Find labels that frequently co-occur with a given label
        
        Args:
            label: Base label
            limit: Maximum results
            
        Returns:
            Cypher query string
        """
        query = f"""
        MATCH (l1:Label {{name: '{label}'}})<-[:HAS_LABEL]-(q:Question)-[:HAS_LABEL]->(l2:Label)
        WHERE l1 <> l2
        WITH l2, count(*) as co_occurrence
        RETURN l2.name as label, co_occurrence
        ORDER BY co_occurrence DESC
        LIMIT {limit}
        """
        
        return query.strip()
    
    def generate_multi_hop_query(self,
                                 labels: List[str],
                                 standard: Optional[str] = None,
                                 limit: int = 10) -> str:
        """
        Advanced multi-hop query for complex relationships
        
        Args:
            labels: List of labels
            standard: Optional standard filter
            limit: Maximum results
            
        Returns:
            Cypher query string
        """
        if not labels:
            return self.generate_label_query([], standard=standard, limit=limit)
        
        # For multiple labels, find questions that match ANY label
        # then expand to related questions
        label_match = " OR ".join([f"l.name = '{label}'" for label in labels])
        
        standard_match = ""
        if standard:
            standard_match = f"WHERE s.name = '{standard}'"
        
        query = f"""
        // Find questions matching any input label
        MATCH (q:Question)-[:HAS_LABEL]->(l:Label)
        WHERE {label_match}
        WITH q, collect(DISTINCT l.name) as direct_labels
        
        // Get related questions through shared labels
        OPTIONAL MATCH (q)-[:HAS_LABEL]->(shared:Label)<-[:HAS_LABEL]-(related:Question)
        WHERE q <> related
        
        // Collect results
        WITH DISTINCT COALESCE(related, q) as result_q, 
             CASE WHEN related IS NULL THEN direct_labels ELSE [] END as primary_match
        
        MATCH (result_q)-[:HAS_LABEL]->(result_l:Label)
        MATCH (result_q)-[:BELONGS_TO_STANDARD]->(s:Standard)
        MATCH (result_q)-[:BELONGS_TO_CLAUSE]->(c:Clause)
        {standard_match}
        
        WITH result_q, s, c, collect(DISTINCT result_l.name) as all_labels, primary_match
        
        RETURN result_q.id as id,
               result_q.text as text,
               s.name as standard,
               c.title as clause,
               all_labels as labels,
               size(primary_match) as direct_match_count
        ORDER BY direct_match_count DESC, size(all_labels) DESC
        LIMIT {limit}
        """
        
        return query.strip()
    
    def generate_path_query(self,
                           start_label: str,
                           end_label: str,
                           max_depth: int = 3) -> str:
        """
        Find paths between two labels through questions
        
        Args:
            start_label: Starting label
            end_label: Ending label
            max_depth: Maximum path depth
            
        Returns:
            Cypher query string
        """
        query = f"""
        MATCH path = shortestPath(
            (l1:Label {{name: '{start_label}'}})-[*1..{max_depth}]-(l2:Label {{name: '{end_label}'}})
        )
        RETURN path
        LIMIT 5
        """
        
        return query.strip()


class QueryBuilder:
    """Fluent interface for building Cypher queries"""
    
    def __init__(self):
        self.match_clauses = []
        self.where_clauses = []
        self.return_clause = ""
        self.limit_value = None
        self.order_by = None
    
    def match(self, pattern: str) -> 'QueryBuilder':
        """Add MATCH clause"""
        self.match_clauses.append(f"MATCH {pattern}")
        return self
    
    def where(self, condition: str) -> 'QueryBuilder':
        """Add WHERE condition"""
        self.where_clauses.append(condition)
        return self
    
    def return_fields(self, fields: str) -> 'QueryBuilder':
        """Set RETURN clause"""
        self.return_clause = f"RETURN {fields}"
        return self
    
    def limit(self, n: int) -> 'QueryBuilder':
        """Set LIMIT"""
        self.limit_value = n
        return self
    
    def order(self, field: str, direction: str = "DESC") -> 'QueryBuilder':
        """Set ORDER BY"""
        self.order_by = f"ORDER BY {field} {direction}"
        return self
    
    def build(self) -> str:
        """Build the final query"""
        parts = []
        parts.extend(self.match_clauses)
        
        if self.where_clauses:
            parts.append(f"WHERE {' AND '.join(self.where_clauses)}")
        
        parts.append(self.return_clause)
        
        if self.order_by:
            parts.append(self.order_by)
        
        if self.limit_value:
            parts.append(f"LIMIT {self.limit_value}")
        
        return "\n".join(parts)


if __name__ == "__main__":
    # Test the generator
    print("\n" + "=" * 80)
    print(" Cypher Query Generator Test")
    print("=" * 80)
    
    generator = CypherQueryGenerator()
    
    # Test 1: Label-based query
    print("\n Query 1: Find questions about backup and risk")
    query1 = generator.generate_label_query(
        labels=['backup', 'risk'],
        standard='iso_27001',
        limit=5
    )
    print(query1)
    
    # Test 2: Related questions
    print("\n Query 2: Find related questions")
    query2 = generator.generate_related_questions_query(
        question_id='doc_001',
        limit=5
    )
    print(query2)
    
    # Test 3: Multi-hop
    print("\n Query 3: Multi-hop advanced query")
    query3 = generator.generate_multi_hop_query(
        labels=['backup', 'documentation'],
        standard='iso_27001',
        limit=10
    )
    print(query3)
