# CrewAI Multi-Agent Literature Annotation Framework

This directory contains a sophisticated multi-agent framework built with CrewAI for annotating scientific literature into Reactome pathway model instances. The system employs four specialized agents working together to provide high-quality, validated annotations.

## 🤖 Architecture Overview

The framework consists of four specialized agents:

### 1. **Reactome Curator Agent** 
- **Role**: Converts structured information into valid Reactome data model instances
- **Expertise**: Reactome schema, pathway modeling, molecular interactions
- **Output**: Valid Reactome entities, reactions, and pathway structures

### 2. **Literature Extractor Agent**
- **Role**: Processes scientific papers and extracts relevant molecular information
- **Expertise**: Biomedical text mining, information extraction, evidence evaluation
- **Output**: Structured molecular interactions, pathways, and functional annotations

### 3. **Domain Expert Reviewer Agent**
- **Role**: Validates biological accuracy and quality of generated annotations
- **Expertise**: Molecular biology, pathway mechanisms, experimental evidence
- **Output**: Quality scores, validation reports, improvement recommendations

### 4. **Quality Checker Agent**
- **Role**: Ensures technical compliance and consistency with existing data
- **Expertise**: Database QA, schema validation, integration testing
- **Output**: Technical review, consistency checks, deployment readiness

## 🔧 Key Components

### Core Files

- **`CrewAILiteratureAnnotator.py`** - Main orchestrator managing the 4-phase workflow
- **`ReactomeAgents.py`** - Agent definitions with specialized roles and backstories
- **`ReactomeTasks.py`** - Task definitions for each phase of the annotation pipeline
- **`ReactomeTools.py`** - Agent-specific tools for accessing data and validation

### Phase-Based Workflow

1. **Phase 1: Literature Extraction** - Extract and structure information from papers
2. **Phase 2: Data Model Creation** - Convert to valid Reactome instances 
3. **Phase 3: Expert Review** - Validate biological accuracy and evidence
4. **Phase 4: Quality Assurance** - Technical validation and consistency checking

## 🚀 Quick Start

### Basic Usage

```python
from reactome_llm.CrewAILiteratureAnnotator import CrewAILiteratureAnnotator, AnnotationRequest
from reactome_llm.GenePathwayAnnotator import GenePathwayAnnotator

# Initialize
annotator = GenePathwayAnnotator()
crewai = CrewAILiteratureAnnotator(annotator)

# Create request
request = AnnotationRequest(
    gene="NTN1",
    papers=["25391454", "22982992", "23467207"],
    quality_threshold=0.7
)

# Run annotation
result = await crewai.annotate_literature(request)
```

### REST API Usage

Start the server:
```bash
flask --app reactome_llm/ReactomeLLMRestAPI run --debug
```

Make a request:
```bash
curl -X POST http://localhost:5000/crewai/annotate \
  -H "Content-Type: application/json" \
  -d '{
    "queryGene": "NTN1",
    "numberOfPubmed": 8,
    "qualityThreshold": 0.7,
    "enableFullText": false
  }'
```

## 📊 Input/Output Formats

### AnnotationRequest
```python
@dataclass
class AnnotationRequest:
    gene: str                          # Target gene symbol
    papers: List[str]                  # PMIDs or full-text content
    pathways: Optional[List[str]]      # Target pathways (optional)
    max_papers: int = 8               # Maximum papers to process
    quality_threshold: float = 0.7     # Minimum quality score
    enable_full_text: bool = False    # Process full-text when available
```

### AnnotationResult
```python
@dataclass
class AnnotationResult:
    gene: str                                    # Gene processed
    reactome_instances: List[Dict[str, Any]]     # Generated Reactome instances
    literature_evidence: List[Dict[str, Any]]    # Supporting evidence
    quality_scores: Dict[str, float]             # Quality metrics
    validation_report: Dict[str, Any]            # Expert review
    consistency_check: Dict[str, Any]            # QA results
    processing_metadata: Dict[str, Any]          # Processing details
```

## 🛠️ Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_key
REACTOME_NEO4J_URI=bolt://localhost:7687
REACTOME_NEO4J_USER=neo4j
REACTOME_NEO4J_PWD=password

# Optional
PUBMED_API_KEY=your_ncbi_key
PUBMED_MONGO_URI=mongodb://localhost:27017
PDF_PAPERS_FOLDER=/path/to/papers
```

### Agent Configuration
```python
crewai = CrewAILiteratureAnnotator(
    gene_annotator=annotator,
    model="gpt-4o-mini",      # LLM model
    temperature=0.1,          # Creativity vs consistency
    max_iter=3,              # Maximum task iterations
    verbose=True             # Enable detailed logging
)
```

## 📈 Quality Metrics

The framework provides comprehensive quality assessment:

### Quality Scores (0-1 scale)
- **Biological Accuracy**: Molecular mechanisms correctness
- **Evidence Support**: Literature backing strength
- **Mechanistic Consistency**: Known biology alignment
- **Integration Quality**: Existing data compatibility

### Validation Criteria
- **Approve**: Score ≥ threshold, no critical issues
- **Requires Revision**: Score ≥ 0.5, minor issues only
- **Reject**: Score < 0.5, major inaccuracies

## 🔍 Agent Tools

### Literature Extractor Tools
- `literature_search` - PubMed paper retrieval
- `fulltext_analysis` - Full-text paper processing
- `protein_interactions` - PPI database access
- `evidence_evaluation` - Evidence strength assessment

### Reactome Curator Tools
- `reactome_query` - Existing pathway data access
- `schema_validation` - Reactome schema compliance
- `protein_interactions` - Molecular interaction modeling
- `evidence_evaluation` - Literature support validation

### Domain Reviewer Tools  
- `literature_search` - Evidence cross-referencing
- `reactome_query` - Existing annotation comparison
- `evidence_evaluation` - Evidence quality assessment
- `quality_metrics` - Quality score calculation
- `consistency_check` - Knowledge consistency validation

### Quality Checker Tools
- `schema_validation` - Technical compliance checking
- `consistency_check` - Data integrity validation
- `quality_metrics` - Automated quality assessment
- `reactome_query` - Integration compatibility testing

## 📝 Examples

See [`examples/crewai_annotation_examples.py`](../examples/crewai_annotation_examples.py) for comprehensive usage examples:

- **Basic Annotation** - Simple gene annotation with default settings
- **Targeted Pathways** - Focus on specific biological pathways
- **High Quality Mode** - Strict quality controls with full-text analysis
- **Batch Processing** - Process multiple genes efficiently

## 🚧 Development

### Adding New Agents
1. Define agent in `ReactomeAgents.py` with role, goal, and backstory
2. Create corresponding tools in `ReactomeTools.py`
3. Add agent-specific tasks in `ReactomeTasks.py`
4. Update toolkit to provide agent tools

### Adding New Tools
1. Inherit from `BaseTool` in `ReactomeTools.py`
2. Implement `_run()` method with error handling
3. Add tool to appropriate agent toolkit method
4. Document tool capabilities and expected inputs/outputs

### Testing
```bash
# Run examples
python examples/crewai_annotation_examples.py

# Test REST API
curl -X GET http://localhost:5000/crewai/status
```

## 🔗 Integration

### With Existing Pipeline
The CrewAI framework integrates seamlessly with existing `GenePathwayAnnotator` functionality:

- Leverages existing PubMed retrieval and caching
- Uses established Neo4j queries and data access
- Maintains compatibility with current REST API
- Preserves all existing validation and scoring mechanisms

### REST API Endpoints
- `POST /crewai/annotate` - Multi-agent annotation
- `GET /crewai/status` - System status and configuration
- All existing endpoints remain available

## 📚 Dependencies

### New Dependencies
```bash
pip install crewai crewai-tools
```

### Key Libraries
- **CrewAI**: Multi-agent orchestration framework
- **LangChain**: LLM integration and prompt management  
- **OpenAI**: GPT model access
- **Pydantic**: Data validation and serialization

## 🎯 Benefits Over Single-Agent Approach

1. **Specialization**: Each agent focuses on their area of expertise
2. **Quality**: Multi-stage validation and review process
3. **Scalability**: Parallel processing and task distribution
4. **Modularity**: Easy to modify or extend individual agents
5. **Transparency**: Clear separation of concerns and responsibilities
6. **Reliability**: Multiple validation layers reduce errors
7. **Flexibility**: Adaptive workflow based on intermediate results

## 📄 License

This CrewAI extension maintains the same license as the parent Reactome curator tool project.