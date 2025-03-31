from langchain_core.prompts import ChatPromptTemplate

# Used to summarize a list of pathways that have been annotated in Reactome.
summary_prompt_template = """
Summarize pathways annotated in Reactome for the following gene based only on the following text, which is provided as context, 
describing the most important pathways in Reactome. Write a paragraph with about {total_words} words, indicating the most important reactions 
or interactions that are related to this gene. Also make sure to cite the original context source using a format like this [pathway_name]
for each sentence. The pathway_name is the text before ":" in each paragraph of the context text.

gene: {gene}

context: {text_for_important_reactome_pathways}
"""
summary_prompt = ChatPromptTemplate.from_template(summary_prompt_template)


# Used to summarize an annotated pathway for a gene in Reactome
annotated_pathway_summary_prompt_template = """
The query gene below has been annotated in a pathway described in the text below. The gene's roles in reactions annotated in the pathway
are also described below. Summarize the following text with about {total_words} words, indicating the most important reactions or interactions 
having the query gene invovled. Don't speculate anything that is not in the provided text.

Query gene: {gene}

{annotated_pathway_text}
"""
annotated_pathway_summary_prompt = ChatPromptTemplate.from_template(annotated_pathway_summary_prompt_template)


# Used to summarize a set of annotated pathways for a gene in Reactome
annotated_pathways_summary_prompt_template = """
The gene below has been annotated in multiple pathways described in the context text below in Reactome. Write a summary having about {total_words} words 
with focus on the molecular functions of the gene in these pathways. Use the context text only for the summary and don't speculate anything that is not 
in the text. Make sure to cite the pathway names in the format like this [Pathway_Name] for each sentence. The pathway names are provided in the
context. Write a summary sentence at the end to summarize all results.

gene: {gene}

context: {annotated_pathways_text}
"""
annotated_pathways_summary_prompt = ChatPromptTemplate.from_template(annotated_pathways_summary_prompt_template)


# Used to summarize a set of interacting pathways predicted for a gene.
interacting_pathways_summary_prompt_template = """
The gene below has not been annotated inside the pathways listed below in Reactome, but predicted to be functionally interacting with these pathways with
false discovery rate (FDR) provided by interacting with genes listed below (interacting_genes). Write a summary with about {total_words} words 
about the interacting pathways summarized in the context below. The summary should focus on interacting_genes' roles in reactions as described 
in the text. Use the context text only for the summary and don't repeat the text. Make sure to mention interacting genes by their names in the 
summary text to provide more detailed molecular mechanistic description. Also mention that the interactions between the gene and its interacting
genes are predicted. Make sure to cite the pathway names in the format like this [Pathway_Name] for each sentence. The pathway names are provided in the
context.

gene: {gene}

pathways(name:fdr): {pathways_with_fdr}

interacting_genes: {interacting_partners}

context: {text_for_interacting_pathways}
"""
interacting_pathways_summary_prompt = ChatPromptTemplate.from_template(interacting_pathways_summary_prompt_template)

# Use to summarize one single interacting pathway 
# The following prompt template has been enhanced by ChatGPT
interacting_pathway_summary_prompt_template = """
The query gene {gene} is predicted to be functionally related to the pathway described below through its interactions with the genes {interacting_genes}.

Summarize the provided pathway description in approximately {total_words} words, highlighting the key reactions or interactions that involve these genes. Ensure that:
	•	The summary begins with a sentence explicitly stating that {gene} interacts with {interacting_genes}.
	•	The roles of the interacting genes in the pathway are clearly described.
	•	No speculative information is included—only summarize what is explicitly stated in the provided text.

Input Data:
	•	Query gene: {gene}
	•	Interacting genes in the pathway: {interacting_genes}
	•	Roles of interacting genes in the pathway: {roles_of_genes}
	•	Pathway description: {interacting_pathway_text}

"""
interacting_pathway_summary_prompt = ChatPromptTemplate.from_template(interacting_pathway_summary_prompt_template)


# Used to evaluate if the abstract text is matched to the pathway text. This is used to validate the found match based on semantic 
# similarity. 
abstract_pathway_match_prompt_template = """
You are an expert in biology. Evaluate whether the following sections of text describe the same biological pathway.  

- The first section is labeled **"pathway_text:"**  
- The second section is labeled **"abstract_text:"**  

Assign a similarity score between **1 and 10**, where:  
- **10** indicates a high likelihood that they describe the same pathway.  
- **1** indicates a very low likelihood.  

### **Response Format:**  
score: XX  

### **Text Sections:**  
**pathway_text:** {pathway_text}  
**abstract_text:** {abstract_text} 
"""
abstract_pathway_match_prompt = ChatPromptTemplate.from_template(abstract_pathway_match_prompt_template)


# Used to summarize the abstract text that are matched to an interacting pathway
abstract_summary_prompt_template = """
The text in the abstract section is excerpts of scientific papers' abstracts collected from PubMed and best matched with pathway text below. 
Write a summary of the abstract text with about {total_words} words to highlight the query_gene and its interaction with interacting_genes, so that 
Reactome curators can create reactions based on the original papers. The generated text should be based on the abstract text below only, providing 
evidence showing the possible functions of the query gene in the pathway, {pathway}. Don't speculate and don't mention interacting genes if you cannot 
see them in the abstract text. Don't just list interacting genes if there is no information.

query_gene: {query_gene}

interacting_genes: {interacting_genes}

pathway_text: {pathway_text}

abstract: {abstract_text}
"""
abstract_summary_prompt = ChatPromptTemplate.from_template(abstract_summary_prompt_template)

# Used to summarize multiple abstracts
# TODO: Somehow pathway name cannot be listed as the source. Need some more investigation to make it work. So far, PMID can be listed.
multiple_abstracts_summary_prompt_template = """
Write a summary based only on the context to summarize the functions of the query gene with about {total_words} words. Highlight the query gene and 
its interactions with genes in the interacting_genes list. If you cannot see any genes listed in the interacting genes in the context, it is fine 
not mentioning them. Don't speculate! Make sure to cite the original context sources, which are provided at the start of each paragraph before ": ", 
using a format like this [PMID: 123456]. Don't just list interacting genes if there is no information.

query_gene: {query_gene}

interacting_genes: {interacting_genes}

context: {context}
"""
multiple_abstracts_summary_prompt = ChatPromptTemplate.from_template(multiple_abstracts_summary_prompt_template)


# Used to extract functional relationships that can be annotated in Reactome
relationship_extraction_prompt_template = """
Extract funtional relationships between the query gene specified below and other genes, proteins, or biological concepts from the
following document. Output the relationships in the following format: {query_gene} - relationship_type -> other gene or protein or 
biological concept. If you can find the cellular component, tissue or cell type, or other experimental system related to the extracted
relationships, make sure to list them.

query_gene: {query_gene}

document: {document}
"""
relationship_extraction_prompt = ChatPromptTemplate.from_template(relationship_extraction_prompt_template)


# Used to summarize abstracts collected from protein interaction databases (e.g. IntAct and BioGrid)
protein_interaction_abstracts_summary_prompt_template = """
You are an expert in biology specializing in protein interactions and pathways. Your task is to summarize the following abstracts, which describe sources of protein interactions for the gene ”{query_gene}”.
	•	If the gene ”{query_gene}” is mentioned, highlight its occurrence in the summary.
    •	If any of the interacting genes "{interactors}" is mentioned, highlight them in the summary.
	•	If any text relates to the pathway ”{pathway}”, highlight it as well.
	•	Ensure your summary includes citations for each statement using the format [PMID: XXXXXX].

The abstracts are listed below, each beginning with “PMID XXXXXX:”. Your summary should be concise, ensuring the total word count does not exceed {total_words}.

Context (Abstracts):
{context}
"""
protein_interaction_abstracts_summary_prompt = ChatPromptTemplate.from_template(protein_interaction_abstracts_summary_prompt_template)
