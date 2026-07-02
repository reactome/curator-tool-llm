"""
prompts.py
----------
All Claude prompt templates for the FullTextPDF pipeline.
Import in the notebook or any pipeline script:
 
    from prompts import build_extraction_prompt
"""
 
 
def build_extraction_prompt(gene: str, text_chunk: str) -> str:
    """
    Prompt for extracting ALL biochemical reactions involving `gene` from a
    300-token text chunk sliding across the full paper.
 
    Every chunk is processed — If no reaction is present, Claude returns an empty reactions list.
 
    Output format follows the Reactome ReactionlikeEvent data model:
      https://curator.reactome.org/cgi-bin/classbrowser?DB=gk_central&CLASS=ReactionlikeEvent
 
    ReactionlikeEvent fields per reaction:
      - name             : display name of the reaction
      - reactionType     : transition / omitted / binding / dissociation / blackBoxEvent
      - input            : PhysicalEntity list — reactants/substrates consumed
      - output           : PhysicalEntity list — products produced
      - catalystActivity : enzyme + GO molecular function
      - regulatedBy      : positiveRegulation / negativeRegulation / requirement
      - compartment      : cellular location
      - summation        : Summation instance {text, literatureReference}
      - relationships    : "EntityA - relationship_type -> EntityB" lines

    NOTE: confidence is Claude's SELF-ASSESSED rating (subjective, 0-1).
    Objective accuracy is measured separately via cosine similarity vs Neo4j.
    """
    return f"""You are an expert Reactome biocurator. From the text below, extract ALL biochemical
reactions described, regardless of which gene or protein is the focus.

A biochemical reaction requires:
  - At least two named biological entities (genes, proteins, metabolites)
  - A directional relationship between them (e.g. phosphorylates, binds, cleaves,
    activates, inhibits, ubiquitinates, translocates, recruits)
  - Evidence this is a real molecular event

For each reaction, express it using the Reactome ReactionlikeEvent data model fields,
and also as relationship lines in this exact format:
EntityA - relationship_type -> EntityB

Use precise relationship types (e.g. phosphorylates, binds, ubiquitinates,
cleaves, activates, inhibits, localizes_to, translocates_to).

Return ONLY a JSON object, no markdown:
{{
  "reactions": [
    {{
      "name": "<display name, e.g. 'PINK1 phosphorylates Parkin at Ser65'>",
      "reactionType": "<transition | omitted | binding | dissociation | blackBoxEvent>",
      "input": ["<PhysicalEntity consumed — reactant or substrate>"],
      "output": ["<PhysicalEntity produced — product>"],
      "catalystActivity": {{
        "catalyst": "<enzyme or catalyst, or null>",
        "molecularFunction": "<GO molecular function term if known, or null>"
      }},
      "regulatedBy": [
        {{
          "regulationType": "<positiveRegulation | negativeRegulation | requirement>",
          "regulator": "<gene, protein, or small molecule>"
        }}
      ],
      "compartment": "<cellular compartment, or null>",
      "summation": {{
        "text": "<factual description of the reaction drawn directly from the text>",
        "literatureReference": ["<PMID or citation string if mentioned, else empty list>"]
      }},
      "relationships": ["EntityA - relationship_type -> EntityB"],
      "confidence": <float 0-1, confidence this reaction is correct and well-supported>
    }}
  ]
}}

If no biochemical reaction is present in this text, return {{"reactions": []}}.

Don't speculate or just list interacting genes if there is no information.

Text:
{text_chunk}"""