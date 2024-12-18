# Use this script to test the RESTful API implemented in ReactomeLLMTestAPI.py. This script basically can be served to check
# the setup and running of the Python virutal env.

import logging
import asyncio
from ReactomeLLMRestAPI import query_gene

logging.basicConfig(level=logging.DEBUG)

async def test_query_gene():
    gene = 'ALDOB'
    result = await query_gene(gene)
    print(result)

asyncio.run(test_query_gene())
