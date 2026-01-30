#!/usr/bin/env python3
"""
Demo avec Knowledge Graph + LLM GRATUIT (Ollama)
Approche 2: RAG basé sur Neo4j et relations sémantiques
"""

import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_graph.graph_builder import ISOGraphBuilder
from src.rag_graph.graph_retriever import GraphRetriever
from src.llm.llm_interface import OllamaLLM

print("=" * 80)
print("DEMO RAG avec KNOWLEDGE GRAPH + LLM GRATUIT")
print("Approche 2: Neo4j + Cypher + Ollama")
print("=" * 80)

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Vérifier Ollama
print("\n[1/6] Vérification d'Ollama...")
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("OK - Ollama disponible")
    else:
        print("ERREUR - Ollama non disponible")
        sys.exit(1)
except:
    print("ERREUR - Ollama non installé")
    sys.exit(1)

# Vérifier Neo4j
print("\n[2/6] Vérification de Neo4j...")
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    driver.close()
    print("OK - Neo4j disponible")
except Exception as e:
    print(f"ERREUR - Neo4j non disponible: {e}")
    print("\nPour démarrer Neo4j avec Docker:")
    print("  docker run -d --name neo4j \\")
    print("    -p 7474:7474 -p 7687:7687 \\")
    print("    -e NEO4J_AUTH=neo4j/password \\")
    print("    neo4j:latest")
    print("\nPuis attendez ~30 secondes et relancez ce script.")
    sys.exit(1)

# Construire le graphe (si nécessaire)
print("\n[3/6] Construction du Knowledge Graph...")
try:
    builder = ISOGraphBuilder(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    
    # Vérifier si le graphe existe déjà
    stats = builder._get_statistics()
    questions_count = stats.get('questions', 0) if stats else 0
    
    if questions_count == 0:
        print("    Graphe vide, construction en cours...")
        builder.build_graph(method=1)
    else:
        print(f"   OK - Graphe existant avec {stats['questions']} questions")
        print(f"      Labels: {stats['labels']}, Standards: {stats['standards']}")
    
    builder.close()
except Exception as e:
    print(f"ERREUR - Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialiser le retriever
print("\n[4/6] Initialisation du Graph Retriever...")
try:
    retriever = GraphRetriever(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    print("OK - Retriever prêt")
except Exception as e:
    print(f"ERREUR - Erreur: {e}")
    sys.exit(1)

# Initialiser Ollama
print("\n[5/6] Initialisation d'Ollama LLM...")
try:
    llm = OllamaLLM(model="mistral", temperature=0.3)
    print("OK - LLM prêt (Mistral)")
except Exception as e:
    print(f"ERREUR - Erreur: {e}")
    sys.exit(1)

# Test de génération
print("\n[6/6] Test avec Knowledge Graph...")
print("=" * 80)

# Test 1: Requete simple
query1 = "Questions sur les sauvegardes de données"
print(f"\n Requete 1: '{query1}'")
print("\n Recherche dans le Knowledge Graph...")

relevant_docs = retriever.retrieve(query1, top_k=8)
print(f"OK - {len(relevant_docs)} documents trouvés via Cypher\n")

print("📚 Documents sources (via relations dans le graphe):")
for i, doc in enumerate(relevant_docs[:5], 1):
    print(f"  [{i}] {doc['content'][:80]}...")
    print(f"      Labels: {doc['metadata']['labels'][:80]}...")

print("\n🤖 Génération avec Ollama (10-30 secondes)...")
print("-" * 80)

try:
    response1 = llm.generate_with_context(
        query=query1,
        context_docs=relevant_docs[:5]
    )
    
    print("\n RÉPONSE GÉNÉRÉE (Method 2 - Graph):")
    print("=" * 80)
    print(response1)
    print("=" * 80)
except Exception as e:
    print(f"ERREUR - Erreur: {e}")

# Test 2: Requete avec relations complexes
print("\n" + "=" * 80)
query2 = "Politique de sécurité et formation des employés ISO 27001"
print(f"\n Requete 2: '{query2}'")

print("\n Recherche multi-hop dans le graphe...")
relevant_docs2 = retriever.retrieve(query2, top_k=8)
print(f"OK - {len(relevant_docs2)} documents trouvés\n")

print("📚 Relations découvertes:")
for i, doc in enumerate(relevant_docs2[:5], 1):
    print(f"  [{i}] {doc['metadata']['iso_standard']} - {doc['metadata']['title']}")
    print(f"      {doc['content'][:100]}...")

print("\n🤖 Génération avec contexte relationnel...")
print("-" * 80)

try:
    response2 = llm.generate_with_context(
        query=query2,
        context_docs=relevant_docs2[:5]
    )
    
    print("\n RÉPONSE GÉNÉRÉE:")
    print("=" * 80)
    print(response2)
    print("=" * 80)
except Exception as e:
    print(f"Erreur: {e}")

# Statistiques
print("\n" + "=" * 80)
print("STATISTIQUES DU KNOWLEDGE GRAPH")
print("=" * 80)

stats = retriever.get_statistics()
print(f"\nGraphe Neo4j:")
print(f"   • Questions (noeuds): {stats.get('questions', 0)}")
print(f"   • Labels (noeuds): {stats.get('labels', 0)}")
print(f"   • Standards (noeuds): {stats.get('standards', 0)}")
print(f"   • Clauses (noeuds): {stats.get('clauses', 0)}")
print(f"   • Relations (aretes): {stats.get('relationships', 0)}")

retriever.close()

print("\n" + "=" * 80)
print("Demo terminee avec succes!")
print("\nAvantages du Knowledge Graph:")
print("   • Decouvre les relations entre concepts")
print("   • Requetes Cypher puissantes")
print("   • Navigation multi-hop dans le graphe")
print("   • Raisonnement sur les connexions semantiques")
print("\nExplorez le graphe: http://localhost:7474")
print("=" * 80 + "\n")
