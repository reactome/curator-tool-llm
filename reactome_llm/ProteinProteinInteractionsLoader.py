import csv
import logging as log
import time
from typing import Set
from xmlrpc.client import boolean
import pandas as pd
import requests

logger = log.getLogger()
logger.setLevel(log.INFO)
# logger.setLevel(log.DEBUG)
log.basicConfig(
    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s', filename=None)

REACTOME_IDG_FI_API_URL = 'https://idg.reactome.org/idgpairwise/relationships/combinedScoreGenesForTerm/'

class PPILoader:
    def __init__(self):
        self.interactions_dict = None
        self.fis = None

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
    
    def get_interactions(self, query_gene, filter_ppis_with_fi: boolean=False, fi_cutoff: float = 0.8) -> dict:
        """For the timebeing, an fi_cutoff value should be specified so that more FIs can be fetched
        from idg.reactome.org. If fi_cutoff is not specified, the latest version of the released FI network
        will be used. The returned interactions may be different from these two versions.

        Args:
            query_gene (_type_): _description_
            filter_ppis_with_fi (boolean, optional): _description_. Defaults to False.
            fi_cutoff (float, optional): _description_. Defaults to None.

        Returns:
            dict: _description_
        """
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
            if fi_cutoff is None:
                if self.fis is None:
                    logger.info('Loading Reactome FIs...')
                    self.fis = self.load_reactome_fis()
                    logger.info('Done loading.')
                # Do a filter
                ppi_dict_fi_filtered = dict()
                for partner, pmids in ppi_dict.items():
                    fi = f"{min(query_gene, partner)}\t{max(query_gene, partner)}"
                    if fi in self.fis:
                        # Make a copy
                        ppi_dict_fi_filtered[partner] = set(pmids)
                return ppi_dict_fi_filtered
            else: # Use idg.reactome.org. This will be updated in the future to avoid calling that
                fi_df = self.query_fis(query_gene, fi_cutoff=fi_cutoff)
                if fi_df is None or fi_df.empty:
                    return None
                partners = set(fi_df['gene'])
                ppi_dict_fi_filtered = dict()
                for partner, pmids in ppi_dict.items():
                    if partner in partners:
                        # Make a copy
                        ppi_dict_fi_filtered[partner] = set(pmids)
                return ppi_dict_fi_filtered
        return None
    
    def query_fis(self,
                  gene: str,
                  fi_cutoff: float = 0.8) -> pd.DataFrame:
        """Query the reactome's idg RESTful API to get the list of functional interactions.

        Args:
            gene (str): _description_
            fi_cutoff (float, optional): _description_. Defaults to 0.8.
        """
        result = requests.get(url='{}{}'.format(REACTOME_IDG_FI_API_URL, gene))
        json_obj = result.json()
        fi_df = pd.DataFrame({'gene': json_obj.keys(),
                            'score': json_obj.values()})
        fi_df = fi_df[fi_df['score'] > fi_cutoff]
        fi_df.sort_values(by=['score'], ascending=False, inplace=True)
        return fi_df
    
    def load_reactome_fis(self, file_name='resources/interactions/FIsInGene_061424_with_annotations.txt') -> Set[str]:
        fis: Set[str] = set()
        fi_df = pd.read_csv(file_name, sep='\t', index_col=None)
        for _, row in fi_df.iterrows():
            # Gene1 and Gene2 should be sorted already
            fis.add('{}\t{}'.format(row['Gene1'], row['Gene2']))
        return fis



# intact_file = 'resources/interactions/intact_human_head_10.txt'
# intact_file = 'resources/interactions/intact_human.txt'
# biogrid_file = 'resources/interactions/BIOGRID-ORGANISM-Homo_sapiens-4.4.243.tab3.head_10.txt'
# biogrid_file = 'resources/interactions/BIOGRID-ORGANISM-Homo_sapiens-4.4.243.tab3.txt'

# ppi_loader = PPILoader()
# a2m_interactions = ppi_loader.get_interactions('A2M')
# print('Interactions for gene A2M: {}'.format(len(a2m_interactions)))
