"""
A customized pubmed retriever so that we can have a good control to wait for enough time to avoid too
many request error and also provide an API key.
"""

import json
import logging
import logging_config
import os
import types
from typing import Iterator
import urllib
import time
import urllib.error
from langchain_community.retrievers import PubMedRetriever
from langchain_core.documents import Document
from pymongo import MongoClient
import dotenv
dotenv.load_dotenv()
# Make sure to get an api key from https://account.ncbi.nlm.nih.gov/settings/ (need to log in first)
pubmed_api_key = os.getenv('PUBMED_API_KEY')

pubmed_mongo_uri = os.getenv('PUBMED_MONGO_URI')
pubmed_mongo_db = os.getenv('PUBMED_MONGO_DB')
pubmed_mongo_collection = os.getenv('PUBMED_MONGO_COLLECTION')

logger = logging.getLogger(__name__)

class ReactomePubMedRetriever(PubMedRetriever):
    # Required by BaseModel
    db: object = None
    maxdate: str = None

    def lazy_load_docs(self, query: str) -> Iterator[Document]:
        for d in self.lazy_load(query=query):
            if d is None:
                continue
            yield self._dict2document(d)
    
    def lazy_load(self, query: str) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query.
        Return an iterator of dictionaries containing the document metadata.
        """
        # To make maxdate settting work, we have to set mindate. Just set a really old date.
        url = (
            self.base_url_esearch
            + "db=pubmed&term="
            + str(urllib.parse.quote(query))
            + f"&retmode=json&retmax={self.top_k_results}&usehistory=y"
            + ('' if self.maxdate is None else '&mindate=1900/01/01&maxdate={}&datetype=pdat'.format(self.maxdate))
            + '&api_key={}'.format(pubmed_api_key)
        )
        # print('URL: {}'.format(url))
        # Add retry
        retry = 0
        while True:
            try: 
                # Wait a little bit just in case
                time.sleep(self.sleep_time)
                result = urllib.request.urlopen(url)
                text = result.read().decode("utf-8")
                json_text = json.loads(text)

                webenv = json_text["esearchresult"]["webenv"]
                for uid in json_text["esearchresult"]["idlist"]:
                    # yield self.retrieve_article(uid, webenv)
                    yield self.get_abstract_from_mongodb(uid)
                break
            except urllib.error.HTTPError as e:
                if retry < self.max_retry:
                    retry += 1
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    print(  # noqa: T201
                        f"Too Many Requests, "
                        f"waiting for {self.sleep_time * retry:.2f} seconds..."
                    )
                    time.sleep(self.sleep_time * retry)
                else:
                    raise e

    def get_abstract_from_mongodb(self, pmid: str|int) -> str:
        """Get the abstract from MongoDB

        Args:
            pmid (int): the PMID of the abstract

        Returns:
            str: the abstract
        """
        # Cache the connection to increase the query performance
        if self.db is None:
            client = MongoClient(pubmed_mongo_uri)
            db = client[pubmed_mongo_db]
            self.db = db
        collection = self.db[pubmed_mongo_collection]
        logger.debug('Query abstract from mongodb: {}'.format(pmid))
        result = collection.find_one({'pmid': str(pmid)})
        if result:
            # Follow the format from _parse_article:
            return {
                "uid": str(pmid),
                "Summary": result['abstract'],
            }
            # return result['abstract']
        else:
            return None


    def retrieve_article(self, uid: str, webenv: str) -> dict:
        url = (
            self.base_url_efetch
            + "db=pubmed&retmode=xml&id={}".format(uid)
            + "&webenv={}".format(webenv)
            + '&api_key={}'.format(pubmed_api_key)
        )

        retry = 0
        while True:
            try:
                time.sleep(self.sleep_time)
                result = urllib.request.urlopen(url)
                #print(url)
                break
            except urllib.error.HTTPError as e:
                # if e.code == 429 and retry < self.max_retry:
                # Updated: Whatever it is
                if retry < self.max_retry:
                    retry += 1
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    print(  # noqa: T201
                        f"Too Many Requests, "
                        f"waiting for {self.sleep_time * retry:.2f} seconds..."
                    )
                    time.sleep(self.sleep_time * retry)
                else:
                    raise e

        xml_text = result.read().decode("utf-8")
        text_dict = self.parse(xml_text)
        return self._parse_article(uid, text_dict)


# Just a simple test
# retriever = ReactomePubMedRetriever()
# pmid = 37941124
# time1 = time.time()
# for i in range(1):
#     print(retriever.get_abstract_from_mongodb(pmid))
# time2 = time.time()
# print('Time: {}'.format(time2 - time1))
