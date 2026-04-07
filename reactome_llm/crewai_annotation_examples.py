"""
Example usage of the CrewAI Multi-Agent Reactome Literature Annotation Framework

This script demonstrates how to use the multi-agent system to annotate literature
into Reactome pathway model instances. It shows both simple and advanced usage
patterns for different types of annotation requests.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure environment is set up
from dotenv import load_dotenv
load_dotenv()

from GenePathwayAnnotator import GenePathwayAnnotator
from CrewAILiteratureAnnotator import CrewAILiteratureAnnotator, AnnotationRequest
from ModelConfig import create_reactome_chat_model, get_crewai_model_settings


def build_annotators(crewai_temperature: float = None, max_iter: int = 3, verbose: bool = True):
    """Create base and CrewAI annotators using centralized model configuration."""
    base_model = create_reactome_chat_model()
    crewai_model_name, default_crewai_temperature = get_crewai_model_settings()

    annotator = GenePathwayAnnotator()
    annotator.set_model(base_model)

    resolved_crewai_temperature = (
        default_crewai_temperature if crewai_temperature is None else crewai_temperature
    )
    crewai_annotator = CrewAILiteratureAnnotator(
        annotator,
        model=crewai_model_name,
        temperature=resolved_crewai_temperature,
        max_iter=max_iter,
        verbose=verbose,
    )
    return annotator, crewai_annotator


async def example_basic_annotation():
    """Basic example: Annotate a well-known gene with default parameters"""
    print("=== Basic Annotation Example ===")
    
    # Initialize components
    _, crewai_annotator = build_annotators(verbose=True)
    
    # Create annotation request
    request = AnnotationRequest(
        gene="TANC1",  # Well-studied gene
        papers=["28754924", "39737163"],  # Sample PMIDs
        max_papers=5,
        quality_threshold=0.7
    )
    
    # Run annotation
    try:
        result = await crewai_annotator.annotate_literature(request)
        
        print(f"Annotation completed for {result.gene}")
        print(f"Reactome instances created: {len(result.reactome_instances)}")
        print(f"Literature evidence: {len(result.literature_evidence)}")
        print(f"Overall quality score: {result.quality_scores.get('overall_quality', 'N/A')}")
        
        # Export results
        output_path = f"results/basic_annotation_{result.gene}.json"
        crewai_annotator.export_results(result, output_path)
        print(f"Results exported to: {output_path}")
        
    except Exception as e:
        print(f"Annotation failed: {str(e)}")


async def example_targeted_pathway_annotation():
    """Advanced example: Focus annotation on specific pathways"""
    print("\n=== Targeted Pathway Annotation Example ===")
    
    # Initialize components
    _, crewai_annotator = build_annotators(verbose=True)
    
    # Create targeted annotation request
    request = AnnotationRequest(
        gene="TP53",  # Well-studied tumor suppressor
        papers=["24051439", "21746913", "23604120"],  # Cancer-related papers
        pathways=["Apoptosis", "DNA damage response", "Cell cycle"],  # Target pathways
        max_papers=8,
        quality_threshold=0.8,  # Higher quality threshold
        enable_full_text=False
    )
    
    try:
        result = await crewai_annotator.annotate_literature(request)
        
        print(f"Targeted annotation completed for {result.gene}")
        print(f"Target pathways: {request.pathways}")
        print(f"Processing metadata: {result.processing_metadata}")
        print("Quality breakdown:")
        for metric, score in result.quality_scores.items():
            print(f"  {metric}: {score}")
        
        # Export results
        output_path = f"results/targeted_annotation_{result.gene}.json"
        crewai_annotator.export_results(result, output_path)
        print(f"Results exported to: {output_path}")
        
    except Exception as e:
        print(f"Targeted annotation failed: {str(e)}")


async def example_high_quality_annotation():
    """Example with strict quality controls and full-text analysis"""
    print("\n=== High Quality Annotation Example ===")
    
    # Initialize components with higher quality settings
    _, crewai_annotator = build_annotators(
        crewai_temperature=0.05,
        max_iter=5,
        verbose=True,
    )
    
    # Create high-quality annotation request
    request = AnnotationRequest(
        gene="BRCA1",  # Important gene with extensive literature
        papers=["25637446", "24025834", "23523072"],  # High-quality cancer papers
        max_papers=6,
        quality_threshold=0.9,  # Very strict quality
        enable_full_text=True  # Enable full-text analysis if available
    )
    
    try:
        result = await crewai_annotator.annotate_literature(request)
        
        print(f"High-quality annotation completed for {result.gene}")
        print(f"Quality scores: {result.quality_scores}")
        print(f"Validation report summary:")
        print(f"  Status: {result.validation_report.get('approval_status', 'unknown')}")
        print(f"Consistency check: {result.consistency_check}")
        
        # Check if quality threshold was met
        overall_quality = result.quality_scores.get('overall_quality', 0)
        if overall_quality >= request.quality_threshold:
            print(f"✓ Quality threshold met ({overall_quality:.2f} >= {request.quality_threshold})")
        else:
            print(f"⚠ Quality threshold not met ({overall_quality:.2f} < {request.quality_threshold})")
        
        # Export results
        output_path = f"results/high_quality_annotation_{result.gene}.json"
        crewai_annotator.export_results(result, output_path)
        print(f"Results exported to: {output_path}")
        
    except Exception as e:
        print(f"High-quality annotation failed: {str(e)}")


async def example_batch_annotation():
    """Example: Batch process multiple genes"""
    print("\n=== Batch Annotation Example ===")
    
    # Initialize components
    _, crewai_annotator = build_annotators(verbose=False)  # Less verbose for batch
    
    # Define batch of genes to process
    genes_to_process = [
        {
            "gene": "NTN1",
            "papers": ["25391454", "22982992"],
            "focus": "axon guidance"
        },
        {
            "gene": "CHD1",
            "papers": ["29249293", "27304074"],
            "focus": "chromatin remodeling"
        },
        {
            "gene": "TANC1",
            "papers": ["27989441", "24751536"],
            "focus": "synaptic function"
        }
    ]
    
    results = []
    
    for gene_info in genes_to_process:
        try:
            request = AnnotationRequest(
                gene=gene_info["gene"],
                papers=gene_info["papers"],
                max_papers=4,  # Smaller for batch processing
                quality_threshold=0.6
            )
            
            print(f"Processing {gene_info['gene']} ({gene_info['focus']})...")
            result = await crewai_annotator.annotate_literature(request)
            results.append(result)
            
            # Export individual results
            output_path = f"results/batch_{result.gene}.json"
            crewai_annotator.export_results(result, output_path)
            
        except Exception as e:
            print(f"Failed to process {gene_info['gene']}: {str(e)}")
    
    # Create batch summary
    print(f"\nBatch processing completed: {len(results)}/{len(genes_to_process)} genes successful")
    
    batch_summary = {
        "total_genes": len(genes_to_process),
        "successful": len(results),
        "results": [
            {
                "gene": r.gene,
                "quality_score": r.quality_scores.get('overall_quality', 0),
                "instances_created": len(r.reactome_instances),
                "evidence_count": len(r.literature_evidence)
            }
            for r in results
        ]
    }
    
    # Export batch summary
    with open("results/batch_summary.json", "w") as f:
        json.dump(batch_summary, f, indent=2)
    
    print("Batch summary exported to: results/batch_summary.json")


def setup_output_directory():
    """Create results directory if it doesn't exist"""
    Path("results").mkdir(exist_ok=True)


async def main():
    """Run all examples"""
    print("CrewAI Multi-Agent Reactome Literature Annotation Examples")
    print("=" * 60)
    
    # Setup
    setup_output_directory()
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    if not os.getenv('REACTOME_NEO4J_URI'):
        print("Error: Reactome Neo4j environment variables not set")
        return
    
    # Run examples
    await example_basic_annotation()
    # await example_targeted_pathway_annotation()
    # await example_high_quality_annotation()
    # await example_batch_annotation()
    
    print("\n" + "=" * 60)
    print("All examples completed! Check the 'results/' directory for outputs.")
    print("\nTo use the REST API, run:")
    print("  flask --app reactome_llm/ReactomeLLMRestAPI run")
    print("Then POST to /crewai/annotate with your gene annotation request.")


if __name__ == "__main__":
    asyncio.run(main())