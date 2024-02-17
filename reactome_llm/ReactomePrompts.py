from langchain_core.prompts import ChatPromptTemplate

# Used to summarize a list of pathways that have been annotated in Reactome.
summary_prompt_template = """
Summarize pathways annotated in Reactome for the following gene based only on the following text, which is provided as context, 
describing the most important pathways in Reactome. Write a paragraph with about {total_words} words, indicating the most important reactions 
or interactions that are related to this gene. Also make sure cite the original context source using a format like this [pathway_name]
for each sentence. The pathway_name is the text before ":" in each paragraph of the context text.

gene: {gene}

context: {text_for_important_reactome_pathways}
"""
summary_prompt = ChatPromptTemplate.from_template(summary_prompt_template)

unannotated_gene_prompt_tempalte = """
The gene below has not been annotated in Reactome, but predicted to be functionally interacting with the following list of pathways with
false discovery rate (FDR) provided by interacting with genes listed below (interacting_genes). Write a summary with about {total_words} words 
about the interacting pathways summarized in the context below. The summary should focus on interacting_genes' roles in reactions as described 
in the text. Use the context text only for the summary and don't repeat the text. Make sure to mention interacting genes by their names in the 
summary text to provide more detailed molecular mechanistic description. Also mention that the interactions between the gene and its interacting
genes are predicted.

gene: {gene}

pathways(name:fdr): {pathways_with_fdr}

interacting_genes: {interacting_partners}

context: {text_for_interacting_pathways}
"""
unannotated_gene_prompt = ChatPromptTemplate.from_template(unannotated_gene_prompt_tempalte)

# Use to summarize the one single interacting pathway 
interacting_pathway_summary_prompt_template = """
The query gene below is predicted to be functionally related with the pathway described below via interacting genes listed also below.
Summarize the following text with about {total_words} words, indicating the most important reactions or interactions having interacting
genes invovled. Make sure the summary has a sentence at the beging say something like gene (i.e. the query gene) is predicted to interact
with a gene (e.g. one of the gene listed in interacting genes).

Query gene: {gene}

{interacting_pathway_text}
"""
interacting_pathway_summary_prompt = ChatPromptTemplate.from_template(interacting_pathway_summary_prompt_template)