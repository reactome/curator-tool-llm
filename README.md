# Reactome Curator Tool - LLM-Powered Gene Pathway Annotation

An intelligent gene pathway annotation system that leverages Large Language Models and multi-agent frameworks to assist Reactome curators in annotating genes and their pathway involvement based on scientific literature.

## 🚀 Features

- **Intelligent Literature Mining**: Automated PubMed abstract retrieval and analysis
- **Multi-Agent Architecture**: CrewAI-powered specialist agents for different annotation tasks
- **Reactome Integration**: Direct integration with Reactome Neo4j database and data models
- **Evidence-Based Annotation**: Literature-supported pathway predictions with confidence scoring
- **Full-Text Analysis**: PDF paper processing for deeper information extraction
- **REST API**: Complete API for programmatic access and integration
- **Interactive Chat Interface**: Chainlit-powered conversational interface

## 🏗️ Architecture

### Traditional Single-Agent Pipeline
- **GenePathwayAnnotator**: Core annotation engine with PubMed integration
- **Literature Processing**: Automated abstract retrieval, embedding, and similarity scoring
- **Pathway Enrichment**: Statistical analysis of protein-protein interactions
- **Reactome Modeling**: Direct pathway instance generation

### New Multi-Agent Framework (CrewAI)
- **ReactomeCurator**: Converts structured data to Reactome instances
- **LiteratureExtractor**: Processes papers and extracts molecular information  
- **Reviewer**: Domain expert validation and quality assessment
- **QualityChecker**: Technical compliance and consistency validation

## 📦 Installation

### Prerequisites
- Python 3.10+
- Neo4j database (Reactome instance)
- MongoDB (for PubMed caching)
- OpenAI API access

### Dependencies
```bash
# Install base dependencies
pip install -r requirements.txt

# Install CrewAI for multi-agent framework
pip install crewai crewai-tools
```
**Note**: At the local mac, use the paperqa env, which has installed all dependencies. 

### Environment Setup
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
PUBMED_API_KEY=your_ncbi_api_key
REACTOME_NEO4J_URI=bolt://localhost:7687
REACTOME_NEO4J_USER=neo4j
REACTOME_NEO4J_PWD=your_password
REACTOME_NEO4J_DATABASE=reactome
PUBMED_MONGO_URI=mongodb://localhost:27017
PUBMED_MONGO_DB=pubmed_cache
PUBMED_MONGO_COLLECTION=abstracts
PDF_PAPERS_FOLDER=./data/papers
```

## 🔧 Usage

### Multi-Agent Annotation (CrewAI)
```python
from reactome_llm.CrewAILiteratureAnnotator import CrewAILiteratureAnnotator, AnnotationRequest
from reactome_llm.GenePathwayAnnotator import GenePathwayAnnotator

# Initialize
annotator = GenePathwayAnnotator() 
crewai = CrewAILiteratureAnnotator(annotator)

# Create annotation request
request = AnnotationRequest(
    gene="NTN1",
    papers=["25391454", "22982992", "23467207"],
    quality_threshold=0.7,
    enable_full_text=False
)

# Run multi-agent annotation
result = await crewai.annotate_literature(request)
print(f"Quality score: {result.quality_scores}")
```

### Traditional Single-Agent Annotation
```python  
from reactome_llm.GenePathwayAnnotator import GenePathwayAnnotator

annotator = GenePathwayAnnotator()
result = await annotator.write_summary_for_gene_annotation("NTN1")
```

### REST API

Start the server:
```bash
flask --app reactome_llm/ReactomeLLMRestAPI run --debug
```

#### Endpoints

**Multi-Agent Annotation:**
```bash
curl -X POST http://localhost:5000/crewai/annotate \
  -H "Content-Type: application/json" \
  -d '{
    "queryGene": "NTN1",
    "numberOfPubmed": 8,
    "qualityThreshold": 0.7,
    "targetPathways": ["Axon guidance"],
    "enableFullText": false
  }'
```

**Traditional Annotation:**
```bash
curl -X POST http://localhost:5000/annotate \
  -H "Content-Type: application/json" \
  -d '{
    "queryGene": "NTN1",
    "numberOfPubmed": 8,
    "cosineSimilarityCutoff": 0.38,
    "llmScoreCutoff": 3
  }'
```

**System Status:**
```bash
curl http://localhost:5000/crewai/status
```

## 📊 Quality Assessment

The multi-agent framework provides comprehensive quality metrics:

- **Biological Accuracy** (0-1): Correctness of molecular mechanisms
- **Evidence Support** (0-1): Strength of literature backing  
- **Mechanistic Consistency** (0-1): Alignment with known biology
- **Integration Quality** (0-1): Compatibility with existing data

### Quality Thresholds
- **Approve**: Score ≥ 0.7, no critical issues
- **Requires Revision**: Score 0.5-0.7, minor issues
- **Reject**: Score < 0.5, major inaccuracies

## 🧪 Testing

Run the validation suite:
```bash
python test_crewai_framework.py
```

Run example workflows:
```bash
python examples/crewai_annotation_examples.py
```

## 🗂️ File Structure

```
reactome_llm/
├── CrewAILiteratureAnnotator.py    # Main multi-agent orchestrator
├── ReactomeAgents.py               # Specialized agent definitions  
├── ReactomeTasks.py                # Task definitions for each phase
├── ReactomeTools.py                # Agent-specific tools
├── GenePathwayAnnotator.py         # Core annotation engine
├── ReactomeLLMRestAPI.py           # REST API with both approaches
├── ReactomeNeo4jUtils.py           # Neo4j database utilities
├── ReactomePubMed.py               # PubMed integration
└── README_CrewAI.md               # Detailed CrewAI documentation

examples/
└── crewai_annotation_examples.py  # Usage examples

test/
└── test_crewai_framework.py       # Validation tests
```

## 🔗 Data Sources

- **Reactome Database**: Neo4j graph database with pathway knowledge
- **PubMed**: Literature abstracts via NCBI E-utilities API  
- **IntAct**: Protein-protein interaction data
- **BioGRID**: Molecular interaction database
- **MongoDB**: Local caching of PubMed abstracts

## 🚀 Deployment

### Server Deployment
To deploy to production server (curator.reactome.org):

1. Zip the reactome_llm folder
2. Transfer and unzip on server
3. Configure .env with production settings
4. Run using the shell script:
```bash
./run_llm.sh
```

To stop the application:
```bash
ps aux | grep llm
kill <process_id>
```

### Database Migration
MongoDB databases are generated locally and migrated:
```bash
# Export from local
mongodump --db your_database_name --out /path/to/backup

# Import to server  
mongorestore --db your_database_name /path/to/backup/your_database_name
```

## 📚 Documentation

- **[CrewAI Framework Guide](reactome_llm/README_CrewAI.md)** - Detailed multi-agent documentation
- **[API Reference](docs/api-reference.md)** - Complete REST API documentation  
- **[Examples](examples/)** - Usage examples and tutorials

## 🤝 Contributing

This tool is part of the Reactome project. For contributions:

1. Follow existing code patterns
2. Add tests for new functionality
3. Update documentation
4. Ensure compatibility with both single and multi-agent approaches

## 📄 License

This project is part of the Reactome curation tools and follows the same licensing terms.
