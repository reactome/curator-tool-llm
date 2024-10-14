"""
A customized pubmed retriever so that we can have a good control to wait for enough time to avoid too
many request error and also provide an API key.
"""

import json
import os
import types
from typing import Iterator
import urllib
import time
import urllib.error
from langchain_community.retrievers import PubMedRetriever
import dotenv
dotenv.load_dotenv()
# Make sure to get an api key from https://account.ncbi.nlm.nih.gov/settings/ (need to log in first)
pubmed_api_key = os.getenv('PUBMED_API_KEY')

class ReactomePubMedRetriever(PubMedRetriever):
    
    def lazy_load(self, query: str) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query.
        Return an iterator of dictionaries containing the document metadata.
        """
        url = (
            self.base_url_esearch
            + "db=pubmed&term="
            + str({urllib.parse.quote(query)})
            + f"&retmode=json&retmax={self.top_k_results}&usehistory=y"
            + '&api_key={}'.format(pubmed_api_key)
        )
        #print('URL: {}'.format(url))
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
                    yield self.retrieve_article(uid, webenv)
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

