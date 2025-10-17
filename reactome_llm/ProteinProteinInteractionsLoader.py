import csv
from json import load
import logging as log
from pprint import pp
import time
from typing import Set
from xmlrpc.client import boolean
import pandas as pd
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import os

logger = log.getLogger()
logger.setLevel(log.INFO)
# logger.setLevel(log.DEBUG)
log.basicConfig(
    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s', filename=None)

REACTOME_IDG_FI_API_URL = 'https://idg.reactome.org/idgpairwise/relationships/combinedScoreGenesForTerm/'

class PPILoader:
    def __init__(self):
        self.interactions_dict = None
        self.fis_dict = None
        self.mongo_fis_loader = None

    def parse_intact_pubmedid(self, publication_ids: str) -> str:
        for token in publication_ids.split('|'):
            if token.startswith('pubmed:'):
                return token.split(':')[1]
        return None

    def parse_intact_gene_name(self, aliases: str) -> str:
        for token in aliases.split('|'):
            if token.endswith('(gene name)') and token.startswith('uniprotkb:'):
                return token.split(':')[1][:-11]
        return None

    def add_interaction_to_dict(self, interactions: dict, gene1: str, gene2: str, pubmedid: str):
        if gene1 not in interactions:
            interactions[gene1] = {}
        if gene2 not in interactions[gene1]:
            interactions[gene1][gene2] = set()
        interactions[gene1][gene2].add(pubmedid)

    def load_intact_interactions(self, intact_file: str = 'resources/interactions/intact_human.txt') -> dict:
        interactions = {}
        with open(intact_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene1 = self.parse_intact_gene_name(row['Alias(es) interactor A'])
                gene2 = self.parse_intact_gene_name(row['Alias(es) interactor B'])
                if gene1 == gene2:
                    continue
                pubmedid = self.parse_intact_pubmedid(row['Publication Identifier(s)'])
                if gene1 and gene2 and pubmedid:
                    self.add_interaction_to_dict(interactions, gene1, gene2, pubmedid)
                    self.add_interaction_to_dict(interactions, gene2, gene1, pubmedid)
        return interactions

    def total_interactions(self, interactions_dict: dict) -> int:
        return sum([len(interactions) for interactions in interactions_dict.values()]) / 2

    def total_pmids(self, interactions_dict: dict) -> int:
        pmids = set()
        for gene, interactions in interactions_dict.items():
            for gene2, pubmedids in interactions.items():
                pmids.update(pubmedids)
        return len(pmids)

    def load_biogrid_interactions(self, file: str = 'resources/interactions/BIOGRID-ORGANISM-Homo_sapiens-4.4.243.tab3.txt') -> dict:
        interactions = {}
        with open(file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene1 = row['Official Symbol Interactor A']
                gene2 = row['Official Symbol Interactor B']
                if gene1 == gene2:
                    continue
                pubmedid = self.parse_biogrid_pubmed(row['Publication Source'])
                if gene1 and gene2 and pubmedid:
                    self.add_interaction_to_dict(interactions, gene1, gene2, pubmedid)
                    self.add_interaction_to_dict(interactions, gene2, gene1, pubmedid)
        return interactions

    def parse_biogrid_pubmed(self, publication_source: str) -> str:
        if publication_source.startswith('PUBMED:'):
            return publication_source.split(':')[1]
        return None

    def load_interactions(self, intact_file: str = 'resources/interactions/intact_human.txt',
                          biogrid_file: str = 'resources/interactions/BIOGRID-ORGANISM-Homo_sapiens-4.4.243.tab3.txt') -> dict:
        """
        Loads and merges protein-protein interactions from IntAct and BioGRID data files.

        This function reads interaction data from two sources: an IntAct file and a BioGRID file.
        It loads the interactions from both sources, then merges them into a single dictionary.
        If the same interaction exists in both sources, their PubMed IDs are combined (union).
        The resulting dictionary maps gene names to their interaction partners and associated PubMed IDs.

        Args:
            intact_file (str): Path to the IntAct interactions file. Defaults to 'resources/interactions/intact_human.txt'.
            biogrid_file (str): Path to the BioGRID interactions file. Defaults to 'resources/interactions/BIOGRID-ORGANISM-Homo_sapiens-4.4.243.tab3.txt'.

        Returns:
            dict: A nested dictionary where the first-level keys are gene names, the second-level keys are interacting gene names,
                    and the values are sets of PubMed IDs supporting the interaction.
        """
        int_ppis = self.load_intact_interactions(intact_file)
        biogrid_ppis = self.load_biogrid_interactions(biogrid_file)
        for gene, interactions in biogrid_ppis.items():
            if gene in int_ppis:
                for gene2, pubmedids in interactions.items():
                    if gene2 in int_ppis[gene]:
                        int_ppis[gene][gene2].update(pubmedids)
                    else:
                        int_ppis[gene][gene2] = pubmedids
            else:
                int_ppis[gene] = interactions
        return int_ppis
    
    def get_interactions(self, 
                         query_gene, 
                         interaction_source: str = 'intact_biogrid',
                         filter_ppis_with_fi: boolean=True, 
                         fi_cutoff: float = 0.8) -> dict:
        """For the time being, an fi_cutoff value should be specified so that more FIs can be fetched
        from idg.reactome.org. If fi_cutoff is not specified, the latest version of the released FI network
        will be used. The returned interactions may be different from these two versions.

        Args:
            query_gene (_type_): _description_
            filter_ppis_with_fi (boolean, optional): _description_. Defaults to True.
            fi_cutoff (float, optional): _description_. Defaults to None.

        Returns:
            dict: _description_
        """
        if interaction_source not in ['intact_biogrid', 'reactome_fis']:
            raise ValueError('interaction_source must be one of intact_biogrid or reactome_fis')
        if interaction_source == 'reactome_fis':
            if self.mongo_fis_loader is None:
                self.mongo_fis_loader = MongoFILoader()
            # Force to use the fi_cutoff value
            if fi_cutoff is None:
                fi_cutoff = 0.8
            fi_df = self.mongo_fis_loader.fetch_fis(query_gene, fi_cutoff=fi_cutoff)
            logger.debug('Total FIs found for gene {} with cutoff {}: {}'.format(query_gene, fi_cutoff, len(fi_df)))
            if fi_df is None or fi_df.empty:
                return dict() # Return an empty dict if no interactions are found to avoid issues in the downstream analysis.
            ppi_dict = dict()
            for _, row in fi_df.iterrows():
                partner = row['gene']
                ppi_dict[partner] = set() # No PubMed IDs are associated with these interactions
            logger.debug('Total PPIs from Reactome FIs for gene {}: {}'.format(query_gene, len(ppi_dict)))
            return ppi_dict
        # This should be the default option
        if self.interactions_dict is None:
            logger.info('Loading PPIs...')
            time_start = time.time()
            self.interactions_dict = self.load_interactions()
            time_end = time.time()
            logger.info(('Time to load interactions: {:.2f} seconds'.format(time_end - time_start)))
        if query_gene in self.interactions_dict.keys():
            ppi_dict = self.interactions_dict[query_gene]
            if not filter_ppis_with_fi:
                return ppi_dict
            if fi_cutoff is None: # Force to use a cutoff value
                fi_cutoff = 0.8
                
            fi_df = self.query_fis(query_gene, fi_cutoff=fi_cutoff)
            logger.debug('Total FIs found for gene {} with cutoff {}: {}'.format(query_gene, fi_cutoff, len(fi_df)))
            if fi_df is None or fi_df.empty:
                return dict() # Return an empty dict if no interactions are found to avoid issues in the downstream analysis.
            partners = set(fi_df['gene'])
            ppi_dict_fi_filtered = dict()
            for partner, pmids in ppi_dict.items():
                if partner in partners:
                    # Make a copy
                    ppi_dict_fi_filtered[partner] = set(pmids)
            logger.debug('Total PPIs after FI filtering for gene {}: {}'.format(query_gene, len(ppi_dict_fi_filtered)))
            return ppi_dict_fi_filtered
        logger.debug('No interactions are found for gene {}'.format(query_gene))
        return dict() # Return an empty dict if no interactions are found to avoid issues in the downstream analysis.
    
    def query_fis(self,
                  gene: str,
                  fi_cutoff: float = 0.8) -> pd.DataFrame:
        """Query the loaded Reactome FIs to get the list of functional interactions above a cutoff.

        Args:
            gene (str): Gene symbol to query.
            fi_cutoff (float, optional): Score threshold. Defaults to 0.8.
        """
        if self.mongo_fis_loader is None:
            self.mongo_fis_loader = MongoFILoader()
        return self.mongo_fis_loader.fetch_fis(gene, fi_cutoff=fi_cutoff)

class MongoFILoader:
    def __init__(self):
        load_dotenv()
        mongo_uri = os.getenv("PUBMED_MONGO_URI")
        # Use FIS-prefixed config for DB and collection names
        mongo_db = os.getenv("FIS_MONGO_DB")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[mongo_db]
        self.gene_index = os.getenv("FIS_MONGO_GENE_INDEX")
        self.relationships = os.getenv("FIS_MONGO_RELATIONSHIPS")
        self.gene_index_dict = self.load_gene_index(collection_name=self.gene_index)

    def load_gene_index(self, collection_name="gene_index"):
        """
        Loads gene index from the specified MongoDB collection.
        Returns a dictionary mapping gene symbols to their indices or info.
        """
        collection = self.db[collection_name]
        gene_index = {}
        doc = collection.find_one()
        if doc:
            for gene_symbol, idx in doc.items():
                gene_index[idx] = gene_symbol
        return gene_index
    
    def fetch_fis(self, query_gene: str, fi_cutoff: float = 0.8) -> pd.DataFrame:
        """
        Fetches functional interactions for a given gene from the MongoDB collection,
        filtering by a specified score cutoff.

        Args:
            query_gene (str): The gene symbol to query.
            fi_cutoff (float): The minimum score threshold for interactions.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered functional interactions.
        """
        collection = self.db[self.relationships]
        relationship_doc = collection.find_one({"_id": query_gene})
        result_df = pd.DataFrame(columns=['gene', 'score'])
        if not relationship_doc or len(relationship_doc) == 0:
            return result_df # Empty dataframe
        row = 0
        for index, score in relationship_doc['combined_score'].items():
            if float(score) > fi_cutoff:
                partner = self.gene_index_dict.get(int(index))
                if partner:
                    result_df.loc[row] = [partner, float(score)]
                    row += 1
        return result_df


# intact_file = 'resources/interactions/intact_human_head_10.txt'
# intact_file = 'resources/interactions/intact_human.txt'
# biogrid_file = 'resources/interactions/BIOGRID-ORGANISM-Homo_sapiens-4.4.243.tab3.head_10.txt'
# biogrid_file = 'resources/interactions/BIOGRID-ORGANISM-Homo_sapiens-4.4.243.tab3.txt'

# ppi_loader = PPILoader()
# reactome_fis_dict = ppi_loader.load_reactome_fis_with_scores()
# gene = 'CHST4'
# gene_partners = reactome_fis_dict[gene]
# for partner, score in gene_partners.items():
#     print('Partner: {}, Score: {}'.format(partner, score))
# print('Total gene partners: {}'.format(len(gene_partners)))

# a2m_interactions = ppi_loader.get_interactions('A2M')
# print('Interactions for gene A2M: {}'.format(len(a2m_interactions)))

# mongo_fis_loader = MongoFILoader()
# gene = 'TANC1'
# gene_interactions = mongo_fis_loader.fetch_fis(gene, fi_cutoff=0.60)
# print(gene_interactions)
