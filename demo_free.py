#!/usr/bin/env python3
"""
Demo Simple avec LLM GRATUIT (Ollama)
Pas besoin de clés API !
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import ISODataLoader
from src.rag_traditional.embeddings import EmbeddingGenerator
from src.rag_traditional.vector_store import VectorStore
from src.rag_traditional.retriever import SemanticRetriever
from src.llm.llm_interface import OllamaLLM

print("=" * 80)
print("DEMO RAG avec LLM GRATUIT (Ollama)")
print("Approche 1: Embeddings + ChromaDB + Mistral")
print("=" * 80)

# Vérifier si Ollama est installé
print("\n[1/7] Vérification d'Ollama...")
import subprocess
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("OK - Ollama est installé et fonctionne")
        print(f"\nModèles disponibles:\n{result.stdout}")
    else:
        print("ERREUR - Ollama n'est pas installé")
        print("\nPour installer Ollama:")
        print("  Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("  Mac: brew install ollama")
        print("\nPuis télécharger un modèle:")
        print("  ollama pull llama3.2")
        sys.exit(1)
except FileNotFoundError:
    print("ERREUR - Ollama n'est pas installé")
    print("\nPour installer Ollama:")
    print("  Linux: curl -fsSL https://ollama.com/install.sh | sh")
    print("  Mac: brew install ollama")
    print("\nPuis télécharger un modèle:")
    print("  ollama pull llama3.2")
    sys.exit(1)

# Charger les données
print("\n[2/7] Chargement des données ISO...")
loader = ISODataLoader()
documents = loader.get_documents_for_rag(method=1)
print(f"OK - {len(documents)} documents chargés")

# Générer les embeddings (on en prend juste 50 pour la démo rapide)
print("\n[3/7] Génération des embeddings (échantillon de 50 documents)...")
embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
sample_docs = documents[:50]  # Échantillon pour démo rapide
sample_docs = embedding_gen.embed_documents(sample_docs)
print(f"OK - Embeddings générés (dimension: {embedding_gen.get_embedding_dimension()})")

# Créer le vector store
print("\n[4/7] Création du vector store...")
vector_store = VectorStore(
    persist_directory="/tmp/iso_demo_free",
    collection_name="iso_demo_free"
)
vector_store.create_collection(reset=True)
vector_store.add_documents(sample_docs)
print(f"OK - {vector_store.get_collection_stats()['document_count']} documents dans le vector store")

# Initialiser le retriever
print("\n[5/7] Initialisation du retriever...")
retriever = SemanticRetriever(
    vector_store=vector_store,
    embedding_generator=embedding_gen
)
print("OK - Retriever prêt")

# Initialiser Ollama LLM
print("\n[6/7] Initialisation d'Ollama LLM (GRATUIT!)...")
try:
    llm = OllamaLLM(model="mistral", temperature=0.3)
    print("OK - Ollama LLM initialisé avec mistral")
except Exception as e:
    print(f"ERREUR - Erreur: {e}")
    print("\nAssurez-vous qu'un modèle est téléchargé:")
    print("  ollama pull llama3.2")
    sys.exit(1)

# Test de génération
print("\n[7/7] Test de génération...")
print("=" * 80)

query = "Questions sur les sauvegardes de données"
print(f"\n Requete: '{query}'")

# Récupérer les documents pertinents
print("\n Recherche des documents pertinents...")
relevant_docs = retriever.retrieve(query, top_k=5)
print(f"OK - {len(relevant_docs)} documents trouvés\n")

print("📚 Documents sources:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"  [{i}] {doc['content'][:80]}...")

# Générer avec Ollama
print("\n🤖 Génération avec Ollama (cela peut prendre 10-30 secondes)...")
print("-" * 80)

try:
    response = llm.generate_with_context(
        query=query,
        context_docs=relevant_docs
    )
    
    print("\nRESULTAT:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    print("\nDemo terminee avec succes!")
    print("\nNote: Ollama est gratuit et fonctionne 100% en local.")
    print("   Pas besoin de cles API, pas de frais, confidentialite totale.")
    
except Exception as e:
    print(f"\nErreur lors de la generation: {e}")
    print("\nAssurez-vous qu'Ollama est demarre:")
    print("  ollama serve")

print("\n" + "=" * 80)
