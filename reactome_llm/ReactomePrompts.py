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
false discovery rate (FDR) provided by interacting with genes listed below (interacting_partners). Write a summary having about {total_words} words 
about these results for this gene. The summary should focus on interacting_parnters in these pathways with information about their annotation. 
The information of these interacting_partners should be extracted from the provided context.

gene: {gene}

pathways(name:fdr): {pathways_with_fdr}

interacting_partners(gene_name:score): {interacting_partners}

context: {text_for_interacting_pathways}
"""

unannotated_gene_prompt = ChatPromptTemplate.from_template(unannotated_gene_prompt_tempalte)
