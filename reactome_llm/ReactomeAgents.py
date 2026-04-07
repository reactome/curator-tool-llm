"""
Specialized Agents for Reactome Literature Annotation

This module defines the 4 core agents that make up the multi-agent framework:
1. ReactomeCurator: Converts structured data into Reactome instances
2. LiteratureExtractor: Processes papers and extracts relevant information  
3. Reviewer: Domain expert validation and quality assessment
4. QualityChecker: Consistency checks and QA compliance

Each agent has specific expertise, tools, and responsibilities in the annotation pipeline.
"""

import logging
from typing import List, Dict, Any, Optional

from crewai import Agent

from ModelConfig import get_crewai_model_settings

logger = logging.getLogger(__name__)


class ReactomeAgents:
    """Factory class for creating specialized Reactome annotation agents"""
    
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        """
        Initialize agent factory with LLM configuration
        
        Args:
            model: OpenAI model to use for all agents (falls back to environment config)
            temperature: Temperature setting for creativity vs consistency (falls back to environment config)
        """
        default_model, default_temperature = get_crewai_model_settings()
        self.model = model or default_model
        self.temperature = default_temperature if temperature is None else temperature
        
    def create_reactome_curator(self, tools: List) -> Agent:
        """
        Create the Reactome Curator agent.
        
        This agent is an expert in the Reactome data model and converts structured 
        information from literature into valid Reactome pathway instances. It has
        deep knowledge of molecular interactions, pathways, and biological processes.
        
        Key responsibilities:
        - Convert literature information to Reactome entities
        - Ensure proper relationships between entities 
        - Create valid pathway representations
        - Maintain data model consistency
        
        Args:
            tools: List of tools available to this agent
            
        Returns:
            Configured Agent instance
        """
        return Agent(
            role="Reactome Data Model Curator",
            goal="Convert structured literature information into valid Reactome pathway model instances",
            backstory="""You are a bioinformatics expert specializing in the Reactome data model.
            
            You have extensive knowledge of:
            - Reactome pathway representation and hierarchy
            - Protein-protein interactions and molecular complexes
            - Biochemical reactions and regulatory mechanisms
            - Gene-pathway relationships and annotations
            - Data model constraints and validation rules
            
            Your expertise allows you to translate literature findings into precise, 
            well-formed Reactome instances that accurately capture molecular mechanisms
            while maintaining semantic consistency with the existing knowledge base.
            
            You are meticulous about data quality and always ensure that created instances
            follow Reactome standards and can be integrated with existing pathway data.""",
            verbose=True,
            allow_delegation=False,
            tools=tools,
            llm=self.model
        )
    
    def create_literature_extractor(self, tools: List) -> Agent:
        """
        Create the Literature Extractor agent.
        
        This agent specializes in reading and analyzing scientific papers to extract
        relevant molecular information. It can process both abstracts and full-text
        papers to identify gene functions, interactions, and pathway involvement.
        
        Key responsibilities:
        - Extract molecular interactions from text
        - Identify pathway involvement and gene functions
        - Summarize findings in structured format
        - Assess evidence strength and reliability
        
        Args:
            tools: List of tools available to this agent
            
        Returns:
            Configured Agent instance
        """
        return Agent(
            role="Scientific Literature Extraction Specialist", 
            goal="Extract and structure molecular information from scientific literature",
            backstory="""You are a computational biologist and literature mining expert.
            
            Your specializations include:
            - Natural language processing for biomedical texts
            - Information extraction from research papers
            - Molecular biology and systems biology knowledge
            - Critical evaluation of experimental evidence
            - Structured data representation of biological findings
            
            You excel at reading complex scientific papers and identifying key molecular
            mechanisms, protein functions, pathway involvement, and experimental evidence.
            You can distinguish between strong and weak evidence, identify contradictions
            in the literature, and extract the most relevant information for pathway
            annotation purposes.
            
            You organize extracted information in a structured, machine-readable format
            that preserves the original evidence and context while enabling downstream
            processing by other specialized agents.""",
            verbose=True,
            allow_delegation=False, 
            tools=tools,
            llm=self.model
        )
    
    def create_reviewer(self, tools: List) -> Agent:
        """
        Create the Domain Expert Reviewer agent.
        
        This agent acts as a senior domain expert who evaluates the quality and 
        accuracy of generated Reactome instances. It has deep biological knowledge
        and can assess whether the annotations make biological sense.
        
        Key responsibilities:
        - Validate biological accuracy of annotations
        - Check consistency with existing knowledge
        - Assess evidence strength and quality
        - Provide improvement recommendations
        
        Args:
            tools: List of tools available to this agent
            
        Returns:
            Configured Agent instance
        """
        return Agent(
            role="Senior Molecular Biology Domain Expert",
            goal="Validate the biological accuracy and quality of generated Reactome annotations", 
            backstory="""You are a distinguished molecular biologist and pathway expert
            with decades of experience in systems biology and pathway curation.
            
            Your expertise spans:
            - Molecular mechanisms and cellular processes
            - Protein function and interaction networks
            - Metabolic and signaling pathways
            - Gene regulation and expression
            - Experimental methods and evidence evaluation
            - Database curation standards and best practices
            
            You have an encyclopedic knowledge of biological pathways and can quickly
            identify when annotations are biologically plausible or problematic. You
            understand the nuances of experimental evidence and can distinguish between
            direct and indirect evidence, high-quality and low-quality studies.
            
            Your role is to act as the final biological authority, ensuring that all
            generated annotations are scientifically sound, well-supported by evidence,
            and consistent with established biological knowledge. You provide constructive
            feedback for improving annotation quality and catching potential errors
            before they enter the knowledge base.""",
            verbose=True,
            allow_delegation=False,
            tools=tools,
            llm=self.model
        )
    
    def create_quality_checker(self, tools: List) -> Agent:
        """
        Create the Quality Checker agent.
        
        This agent ensures that generated instances are consistent with existing
        Reactome data and pass all quality assurance tests. It focuses on technical
        compliance and data integrity rather than biological accuracy.
        
        Key responsibilities:
        - Check data format compliance
        - Validate against existing instances
        - Run QA tests and consistency checks
        - Ensure integration compatibility
        
        Args:
            tools: List of tools available to this agent
            
        Returns:
            Configured Agent instance
        """
        return Agent(
            role="Database Quality Assurance Specialist",
            goal="Ensure generated Reactome instances meet all technical and consistency requirements",
            backstory="""You are a database engineer and quality assurance expert
            specializing in biological databases and knowledge graphs.
            
            Your technical expertise includes:
            - Database schema design and validation
            - Data integrity and consistency checking
            - Quality assurance protocols and testing
            - Integration testing and compatibility
            - Performance optimization and indexing
            - Graph database relationships and constraints
            
            You have intimate knowledge of the Reactome database structure, all QA
            tests, and consistency requirements. You know exactly what technical
            specifications must be met for new instances to be successfully integrated
            without breaking existing functionality.
            
            Your role is to be the technical gatekeeper, ensuring that all generated
            content meets the highest standards for data quality, format compliance,
            and system compatibility. You catch technical issues that could cause
            problems downstream and ensure smooth integration with the existing
            Reactome infrastructure.
            
            You work closely with the domain expert but focus on the technical rather
            than biological aspects of quality assurance.""",
            verbose=True,
            allow_delegation=False,
            tools=tools,
            llm=self.model
        )
    
    def get_all_agents(self, toolkit) -> Dict[str, Agent]:
        """
        Get all agents configured with appropriate tools
        
        Args:
            toolkit: ReactomeToolkit instance providing agent tools
            
        Returns:
            Dictionary mapping agent names to configured agents
        """
        return {
            "curator": self.create_reactome_curator(toolkit.get_curator_tools()),
            "extractor": self.create_literature_extractor(toolkit.get_extractor_tools()),
            "reviewer": self.create_reviewer(toolkit.get_reviewer_tools()),
            "qa_checker": self.create_quality_checker(toolkit.get_qa_tools())
        }