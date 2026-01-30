# Système RAG pour Questionnaires ISO

Génération automatique de questionnaires d'audit ISO avec LLM gratuit (Ollama + Mistral).

## Description

Ce projet propose deux approches pour interroger 1,050 questions provenant de 5 normes ISO :

### Approche 1 : RAG Vectoriel
Recherche par similarité sémantique (embeddings + ChromaDB)

### Approche 2 : Knowledge Graph
Recherche par relations (Neo4j + Cypher + détection de labels)

## Installation

```bash
# Cloner et installer les dépendances
cd /home/khaldoun/prjt_vap
pip install -q ollama sentence-transformers chromadb pandas neo4j

# Vérifier qu'Ollama est installé
ollama --version

# Télécharger le modèle Mistral (si pas déjà fait)
ollama pull mistral
```

---

## Approche 1 : RAG Vectoriel

### Lancement
```bash
python demo_free.py
```

### Fonctionnement
1. Charge 1,050 questions ISO
2. Génère les embeddings (384 dimensions)
3. Stocke dans ChromaDB
4. Recherche sémantique
5. Génération avec Mistral

### Exemple de requête
```
"Questions sur les sauvegardes de données"
→ Trouve 5 questions pertinentes par similarité
→ Génère un questionnaire de 12 questions
```

### Temps d'exécution
- Première fois : environ 60 secondes (création des embeddings)
- Exécutions suivantes : environ 20 secondes

---

## Approche 2 : Knowledge Graph

### Prérequis : Démarrer Neo4j

```bash
# Démarrer Neo4j avec Docker
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Attendre 30 secondes que Neo4j démarre
sleep 30
```

### Lancement
```bash
python demo_graph_free.py
```

### Fonctionnement
1. Vérifie que Neo4j et Ollama sont disponibles
2. Construit le graphe (première fois seulement)
   - 1,050 nœuds Question
   - 580 nœuds Label
   - 5 nœuds Standard
   - 200 nœuds Clause
   - 6,556 relations
3. Détecte les labels dans la requête
4. Génère des requêtes Cypher
5. Recherche multi-hop dans le graphe
6. Génération avec Mistral

### Exemple de requête
```
"Politique de sécurité et formation des employés ISO 27001"
→ Détecte : labels=[policy, training, employee], standard=iso_27001
→ Génère Cypher pour naviguer dans le graphe
→ Trouve 8 questions via relations
→ Génère questionnaire structuré
```

### Temps d'exécution
- Première fois : environ 3 minutes (construction du graphe)
- Exécutions suivantes : environ 30 secondes

### Interface Neo4j
Ouvrir http://localhost:7474 dans un navigateur
- User: `neo4j`
- Password: `password`

**Requête Cypher exemple :**
```cypher
MATCH (q:Question)-[:HAS_LABEL]->(l:Label {name: 'backup'})
RETURN q.text, l.name
LIMIT 5
```

---

## Comparaison des Approches

| Critère | RAG Vectoriel | Knowledge Graph |
|---------|---------------|-----------------||
| **Vitesse** | Très rapide | Rapide |
| **Précision simple** | Excellente | Très bonne |
| **Relations complexes** | Limitée | Excellente |
| **Setup** | Simple | Nécessite Docker |
| **Use Case** | Recherche directe | Analyse relationnelle |

**Recommandations d'utilisation :** 
- Requêtes simples : Approche 1
- Requêtes complexes/multi-critères : Approche 2

## Commandes Utiles

### Approche 1 (Vectoriel)
```bash
# Lancer la démo
python demo_free.py

# Reconstruire l'index ChromaDB
rm -rf /tmp/iso_demo_free && python demo_free.py
```

### Approche 2 (Graph)
```bash
# Démarrer Neo4j
docker start neo4j  # Si déjà créé
# OU
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:latest

# Lancer la démo
python demo_graph_free.py

# Reconstruire le graphe
# Le script détecte automatiquement et reconstruit si vide

# Arrêter Neo4j
docker stop neo4j
```

### Gérer Ollama
```bash
# Lister les modèles
ollama list

# Télécharger un modèle
ollama pull mistral
ollama pull llama3.2

# Tester Ollama
ollama run mistral "Bonjour, réponds en français"
```

## Structure des Données

```
data/
├── method 1/          # 1,050 questions avec labels détaillés
│   ├── labeled_our_iso_27001.csv  (200 questions)
│   ├── labeled_our_iso_27002.csv  (250 questions)
│   ├── labeled_our_iso_27017.csv  (200 questions)
│   ├── labeled_our_iso_27018.csv  (200 questions)
│   └── labeled_our_iso_27701.csv  (200 questions)
└── method 2/          # Labels alternatifs (non utilisé)
```

**Format CSV :**
| Colonne | Description | Exemple |
|---------|-------------|---------|
| `id` | Identifiant unique | `doc_001` |
| `text` | Question ISO | `"Are backups tested regularly?"` |
| `title` | Clause ISO | `"Clause 8 – Operation"` |
| `labels` | Mots-clés | `"backup, testing, documentation"` |

## Utilisation Avancée (API Python)

### Approche 1 : API Vectorielle

```python
from src.rag_traditional import ISORAGSystem

# Initialiser
rag = ISORAGSystem(
    data_method=1,
    embedding_model="all-MiniLM-L6-v2",
    llm_provider="ollama",
    llm_model="mistral",
    rebuild_index=False
)

# Générer un questionnaire
result = rag.generate_questionnaire(
    topic="backup de données",
    iso_standard="iso_27001",
    num_questions=10
)
print(result['answer'])
```

### Approche 2 : API Graph

```python
from src.rag_graph import GraphRetriever
from src.llm import OllamaLLM

# Initialiser
retriever = GraphRetriever(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
llm = OllamaLLM(model="mistral")

# Rechercher
docs = retriever.retrieve("Questions sur les backups", top_k=5)

# Générer
response = llm.generate_with_context(
    query="Créez un questionnaire",
    context_docs=docs
)
print(response)

retriever.close()
```

## Exemples de Requêtes

### Requêtes simples (Approche 1 ou 2)
```
"Questions sur les sauvegardes"
"Évaluation des risques ISO 27001"
"Formation des employés"
```

### Requêtes complexes (Préférer Approche 2)
```
"Politique de sécurité ET formation des employés ISO 27001"
"Backups ET disaster recovery ET restauration"
"Clause 5 avec leadership et gouvernance"
```

### Requêtes Cypher (Neo4j uniquement)
```cypher
// Toutes les questions sur backup
MATCH (q:Question)-[:HAS_LABEL]->(l:Label {name: 'backup'})
RETURN q.text LIMIT 10

// Questions ISO 27001 Clause 5
MATCH (q:Question)-[:BELONGS_TO_STANDARD]->(s:Standard {name: 'iso_27001'})
MATCH (q)-[:BELONGS_TO_CLAUSE]->(c:Clause)
WHERE c.title CONTAINS 'Clause 5'
RETURN q.text, c.title

// Labels les plus utilisés
MATCH (l:Label)<-[:HAS_LABEL]-(q:Question)
RETURN l.name, count(q) as usage
ORDER BY usage DESC LIMIT 10
```

## Structure du Projet

```
prjt_vap/
├── demo_free.py              # Demo Approche 1 (Vectoriel)
├── demo_graph_free.py        # Demo Approche 2 (Graph)
├── data/                     # Données CSV (1,050 questions)
│   └── method 1/
├── src/
│   ├── rag_traditional/      # Approche 1
│   │   ├── embeddings.py     # Sentence Transformers
│   │   ├── vector_store.py   # ChromaDB
│   │   ├── retriever.py      # Recherche sémantique
│   │   └── query_handler.py  # Système complet
│   ├── rag_graph/            # Approche 2
│   │   ├── graph_builder.py  # Construction Neo4j
│   │   ├── label_detector.py # Extraction labels
│   │   ├── cypher_generator.py # Génération requêtes
│   │   └── graph_retriever.py  # Recherche graph
│   ├── llm/                  # Interface LLM
│   │   └── llm_interface.py  # Ollama/OpenAI/Anthropic
│   └── utils/
│       └── data_loader.py    # Chargement CSV
├── notebooks/                # Démos interactives
└── requirements.txt
```

## Technologies Utilisées

### Approche 1
- **Sentence Transformers** : all-MiniLM-L6-v2 (384D)
- **ChromaDB** : Vector store persistant
- **Ollama** : LLM local gratuit (Mistral)

### Approche 2
- **Neo4j** : Base de données graphe
- **Cypher** : Langage de requêtes
- **Détection NLP** : Extraction automatique de labels
- **Ollama** : LLM local gratuit (Mistral)
