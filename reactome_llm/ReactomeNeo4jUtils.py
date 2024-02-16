import os
import dotenv
from neo4j import GraphDatabase
import neo4j

import pandas as pd

# Handle configuration in .env
dotenv.load_dotenv()
URI = os.getenv('REACTOME_NEO4J_URI')
AUTH = (os.getenv('REACTOME_NEO4J_USER'), os.getenv('REACTOME_NEO4J_PWD'))
DB = os.getenv('REACTOME_NEO4J_DATABASE')


def load_event_to_topic_map() -> pd.DataFrame:
    """Load the map from the pathways to their TopLevelPathways.
    Note: Some of pathways may not have any top level pathways.
    Returns:
        pd.DataFrame: _description_
    """
    query = '''
        MATCH (p:Event) <- [:hasEvent*] - (t:TopLevelPathway) 
        RETURN p.dbId as pathway_id, p.displayName as pathway_name,
               t.dbId as topic_id, t.displayName as topic_name
    '''
    top_pathway_df = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        top_pathway_df = driver.execute_query(query,
                                              db=DB,
                                              result_transformer_=neo4j.Result.to_df)
    return top_pathway_df


def load_pathway_dbIds() -> list:
    """Load the list of dbIds for pathways. This may be used to filter events to pathways only.

    Returns:
        list: _description_
    """
    query = 'MATCH (p:Pathway) RETURN p.dbId as dbId'
    pathway_ids = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        pathway_ids = driver.execute_query(query,
                                           db=DB,
                                           result_transformer_=neo4j.Result.to_df)['dbId'].to_list()
    return pathway_ids


def load_event_summary(limit: int = None) -> pd.DataFrame:
    """Load text summary for all pathways and reactions.

    Returns:
        pd.DataFrame: the dataframe contains columns, dbId, displayName, text summary.
    """
    event_summary_pd = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=DB) as session:
            event_summary_pd = session.execute_read(
                _run_event_summary_query, limit)
    # Set dbId as the index
    event_summary_pd.set_index('dbId', drop=False, inplace=True)
    return event_summary_pd


def _run_event_summary_query(tx, limit: int = None) -> pd.DataFrame:
    """Run the actual cyphery query.

    Args:
        tx (_type_): _description_

    Returns:
        pd.DataFrame: the query results in a DataFrame object.
    """
    query = """
        Match (e:Event) 
        OPTIONAL Match (e)-[:summation]->(s:Summation)
        return e.dbId as dbId, e.displayName as displayName, s.text as summary 
    """
    if limit:
        query = (query + ' limit {}').format(limit)
    result = tx.run(query)
    return result.to_df()

