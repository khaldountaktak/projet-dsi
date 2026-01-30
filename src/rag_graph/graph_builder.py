"""
Knowledge Graph Builder for ISO Standards
Constructs a Neo4j graph from CSV data with relationships
"""

from typing import List, Dict, Optional
import logging
from neo4j import GraphDatabase
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import ISODataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ISOGraphBuilder:
    """Build and populate Neo4j knowledge graph from ISO data"""
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password"):
        """
        Initialize Neo4j connection
        
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
            logger.info("\nPour démarrer Neo4j:")
            logger.info("  Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
            logger.info("  Ou installer Neo4j Desktop: https://neo4j.com/download/")
            raise
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("  Database cleared")
    
    def create_constraints(self):
        """Create uniqueness constraints for better performance"""
        constraints = [
            "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT label_name IF NOT EXISTS FOR (l:Label) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT standard_name IF NOT EXISTS FOR (s:Standard) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")
        
        logger.info("OK - Constraints created")
    
    def build_graph(self, method: int = 1):
        """
        Build the complete knowledge graph
        
        Args:
            method: Data method (1 or 2)
        """
        logger.info(f"  Building knowledge graph from method {method} data...")
        
        # Clear existing data
        self.clear_database()
        
        # Create constraints
        self.create_constraints()
        
        # Load data
        loader = ISODataLoader()
        documents = loader.get_documents_for_rag(method=method)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Build graph
        self._create_nodes(documents)
        self._create_relationships(documents)
        
        # Create statistics
        stats = self._get_statistics()
        logger.info("OK - Graph construction complete!")
        logger.info(f" Statistics:")
        logger.info(f"   Questions: {stats['questions']}")
        logger.info(f"   Labels: {stats['labels']}")
        logger.info(f"   Standards: {stats['standards']}")
        logger.info(f"   Clauses: {stats['clauses']}")
        logger.info(f"   Relationships: {stats['relationships']}")
    
    def _create_nodes(self, documents: List[Dict]):
        """Create all nodes in the graph"""
        with self.driver.session() as session:
            for doc in documents:
                metadata = doc['metadata']
                
                # Parse labels
                labels_str = metadata.get('labels', '')
                labels = [l.strip() for l in labels_str.split(',') if l.strip()]
                
                # Create Question node
                session.run("""
                    MERGE (q:Question {id: $id})
                    SET q.text = $text,
                        q.content = $text
                """, id=doc['id'], text=doc['content'])
                
                # Create Standard node
                standard = metadata.get('iso_standard', 'unknown')
                session.run("""
                    MERGE (s:Standard {name: $name})
                """, name=standard)
                
                # Create Clause node
                clause_title = metadata.get('title', 'unknown')
                clause_id = f"{standard}_{clause_title.replace(' ', '_')}"
                session.run("""
                    MERGE (c:Clause {id: $id})
                    SET c.title = $title,
                        c.standard = $standard
                """, id=clause_id, title=clause_title, standard=standard)
                
                # Create Label nodes
                for label in labels:
                    session.run("""
                        MERGE (l:Label {name: $name})
                    """, name=label.lower())
        
        logger.info("OK - Nodes created")
    
    def _create_relationships(self, documents: List[Dict]):
        """Create relationships between nodes"""
        with self.driver.session() as session:
            for doc in documents:
                metadata = doc['metadata']
                
                # Parse labels
                labels_str = metadata.get('labels', '')
                labels = [l.strip().lower() for l in labels_str.split(',') if l.strip()]
                
                standard = metadata.get('iso_standard', 'unknown')
                clause_title = metadata.get('title', 'unknown')
                clause_id = f"{standard}_{clause_title.replace(' ', '_')}"
                
                # Question -> Standard
                session.run("""
                    MATCH (q:Question {id: $qid})
                    MATCH (s:Standard {name: $standard})
                    MERGE (q)-[:BELONGS_TO_STANDARD]->(s)
                """, qid=doc['id'], standard=standard)
                
                # Question -> Clause
                session.run("""
                    MATCH (q:Question {id: $qid})
                    MATCH (c:Clause {id: $cid})
                    MERGE (q)-[:BELONGS_TO_CLAUSE]->(c)
                """, qid=doc['id'], cid=clause_id)
                
                # Clause -> Standard
                session.run("""
                    MATCH (c:Clause {id: $cid})
                    MATCH (s:Standard {name: $standard})
                    MERGE (c)-[:PART_OF_STANDARD]->(s)
                """, cid=clause_id, standard=standard)
                
                # Question -> Labels
                for label in labels:
                    session.run("""
                        MATCH (q:Question {id: $qid})
                        MATCH (l:Label {name: $label})
                        MERGE (q)-[:HAS_LABEL]->(l)
                    """, qid=doc['id'], label=label)
        
        logger.info("OK - Relationships created")
    
    def _get_statistics(self) -> Dict:
        """Get graph statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (q:Question) WITH count(q) as questions
                MATCH (l:Label) WITH questions, count(l) as labels
                MATCH (s:Standard) WITH questions, labels, count(s) as standards
                MATCH (c:Clause) WITH questions, labels, standards, count(c) as clauses
                MATCH ()-[r]->() WITH questions, labels, standards, clauses, count(r) as relationships
                RETURN questions, labels, standards, clauses, relationships
            """)
            
            record = result.single()
            if record:
                return {
                    'questions': record['questions'],
                    'labels': record['labels'],
                    'standards': record['standards'],
                    'clauses': record['clauses'],
                    'relationships': record['relationships']
                }
            else:
                return {
                    'questions': 0,
                    'labels': 0,
                    'standards': 0,
                    'clauses': 0,
                    'relationships': 0
                }
    
    def visualize_sample(self, limit: int = 10):
        """
        Get a sample of the graph for visualization
        
        Args:
            limit: Number of questions to include
            
        Returns:
            Cypher query result
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (q:Question)-[r]->(n)
                RETURN q, r, n
                LIMIT $limit
            """, limit=limit)
            
            return list(result)


def main():
    """Build the knowledge graph"""
    print("=" * 80)
    print("  ISO Knowledge Graph Builder")
    print("=" * 80)
    
    # Connection parameters
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    print(f"\n📡 Connecting to Neo4j at {NEO4J_URI}...")
    
    try:
        builder = ISOGraphBuilder(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        
        # Build graph from method 1 data
        builder.build_graph(method=1)
        
        # Show sample
        print("\n Sample of the graph structure:")
        sample = builder.visualize_sample(limit=5)
        for record in sample[:3]:
            q = record['q']
            r = record['r']
            n = record['n']
            print(f"  ({q['id']}) -[{type(r).__name__}]-> ({dict(n).get('name', dict(n).get('title', 'N/A'))})")
        
        builder.close()
        
        print("\nOK - Graph built successfully!")
        print("\nNote: You can now explore the graph at http://localhost:7474")
        
    except Exception as e:
        print(f"\nERREUR - Error: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")


if __name__ == "__main__":
    main()
