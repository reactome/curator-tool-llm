# This script is used to embed text using SentenceEmbedder.
from sentence_transformers import SentenceTransformer
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from typing import Dict, List, Tuple
import numpy as np
import logging
import gc

logger = logging.getLogger(__name__)

# The lanugage model to be used
# BERT_LANGUAGE_MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# For some reason, the aboe lanaguage model throws error. Use the folowing instead.
BERT_LANGUAGE_MODEL = 'dmis-lab/biobert-v1.1'
# PATHWAY_EMBEDDING_FILE = DIR + "pathway2embedding_biobert_112421.pkl"
# OUT_FILE = DIR + "pathway2embedding_112321.pkl"
# The default model: tried this based model. The performance of this model performs better than the above.
# Very weird!!!
# BERT_LANGUAGE_MODEL = 'bert-base-uncased'
# PATHWAY_EMBEDDING_FILE = DIR + 'pathway2embedding_bert_base_uncased_112421.pkl'
# BERT_LANGUAGE_MODEL = 'bert-large-cased'
# PATHWAY_EMBEDDING_FILE = DIR + 'pathway2embedding_bert_large_cased_112421.pkl'
# BERT_LANGUAGE_MODEL = 'roberta-base'
# PATHWAY_EMBEDDING_FILE = DIR + 'pathway2embedding_robert_base_112421.pkl'
LAYERS = '-1'

# Minimum score to remove something that is not counted
MINIMUM_SCORE = 1.0e-22
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
MAX_SENTENCE_LENGTH = 512

# Cross-encoder for Stage-2 re-ranking. Unlike the bi-encoder (sentence_embed + cosine_similarity),
# which embeds query and abstract independently, the cross-encoder scores the (query, abstract) pair
# jointly -> higher-fidelity relevance. Cheap enough to run on the full pool on CPU (~7.4s / 300
# abstracts), so it replaces the bi-encoder as the re-ranker while the bi-encoder helpers stay for
# other callers.
CROSS_ENCODER_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"
_CROSS_ENCODER: object = None


def _get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL_NAME):
    """Lazily construct and cache the cross-encoder (loading it is expensive; reuse across calls)."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        import torch
        import transformers
        from sentence_transformers import CrossEncoder
        prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            _CROSS_ENCODER = CrossEncoder(model_name)
        finally:
            transformers.logging.set_verbosity(prev_verbosity)
        # MedCPT emits raw relevance LOGITS (e.g. +15 relevant, -15 irrelevant). sentence-transformers
        # defaults a 1-label CrossEncoder to a Sigmoid, which saturates every relevant pair to ~1.0 and
        # destroys the ranking. Force identity so predict() returns the discriminating raw logits.
        _CROSS_ENCODER.activation_fn = torch.nn.Identity()
    return _CROSS_ENCODER


def cross_encoder_rerank(query_text: str, abstracts: list) -> list:
    """Score each abstract dict against `query_text` with the cross-encoder and return them sorted
    by descending relevance. Each dict must carry its abstract text under 'Summary' (the pool-doc
    key used throughout retrieval); the score is written back as 'cross_score'."""
    cross_encoder = _get_cross_encoder()
    pairs = [(query_text, a['Summary']) for a in abstracts]
    scores = cross_encoder.predict(pairs)
    for a, s in zip(abstracts, scores):
        a['cross_score'] = float(s)
    return sorted(abstracts, key=lambda a: a['cross_score'], reverse=True)


def sentence_embed(text: str,
                   embedding_approach: SentenceTransformer = None):
    if embedding_approach is None:
        embedding_approach = create_sentence_transformer()
    return embedding_approach.encode(text)


def sentence_embed_pmid(abstract: str,
                        pmid: str,
                        embedding_approach: SentenceTransformer):
    embedding = sentence_embed(abstract)
    return {pmid: embedding}


def create_sentence_transformer() -> SentenceTransformer:
    logger.info("Create a new SentenceTransformer object...")
    embedding_approach = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    embedding_approach.max_seq_length = MAX_SENTENCE_LENGTH
    return embedding_approach


def generate_sentence_embedding(pathway2doc,
                                embedding_approach: SentenceTransformer = None) -> Dict[str, List[np.ndarray]]:
    logger.info("generate_sentence_embedding for {}...".format(len(pathway2doc)))
    if embedding_approach is None:
        embedding_approach = create_sentence_transformer()
    pathway2embedding = dict()
    counter = 1
    for pathway, doc in pathway2doc.items():
        # logger.info("{}: {}".format(counter, pathway))
        embedding = sentence_embed(doc, embedding_approach)
        pathway2embedding[pathway] = embedding
        counter = counter + 1
        # if counter == 5:
        #     break
    logger.info("The size of pathway2embeding: {}".format(len(pathway2embedding)))
    return pathway2embedding


def bert_embed(document: str,
               language_model: str = BERT_LANGUAGE_MODEL) -> np.ndarray:
    embedding_approach = TransformerDocumentEmbeddings(language_model, layers=LAYERS, layer_mean=True)
    sentence = Sentence(document)
    embedding_approach.embed(sentence)
    return sentence.embedding.detach().numpy()


def generate_bert_embedding(pathway2doc: dict,
                            language_model: str = BERT_LANGUAGE_MODEL) -> dict:
    """
    Generate the pathway to embedding using Flair
    :param pathway2doc
    :param language_model: the pre-trained language model to be used. See the documents
    for possible model that can be used in the Flair API.
    :param save: If anything is provided, the output will be saved
    :return:
    """
    # The same emebeding may be used in multiple batches as long as the sentences are cleaned
    embedding_approach = TransformerDocumentEmbeddings(language_model,
                                                       layer_mean=True,
                                                       layers=LAYERS,
                                                       pooling='cls')  # cls produces the best results
    pathway2embedding = dict()
    # pathway_list = random.sample(list(pathway2doc.keys()), 10)
    # pathway_list = list(pathway2doc.keys())
    # doc_list = [pathway2doc[pathway] for pathway in pathway_list]
    # Want to run embedding sentence by sentence. Tried to use batch, which gives different results.
    # Need to do more reading about how this is implemented.
    counter = 1
    for pathway, doc in pathway2doc.items():
        print("{}: {}".format(counter, pathway))
        sentence = Sentence(doc)
        embedding_approach.embed(sentence)
        pathway2embedding[pathway] = sentence.embedding.detach().numpy()
        # Force gc: https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python
        del sentence
        gc.collect()
        counter = counter + 1
    logging.info("The size of pathway2embeding: {}".format(len(pathway2embedding)))
    return pathway2embedding


def convert_to_matrix(pathway2embedding: dict) -> Tuple[np.ndarray, List[str]]:
    # Convert to a matrix
    embedding_matrix = None
    pathway_list = []
    for pathway in pathway2embedding.keys():
        pathway_list.append(pathway)
        embedding_array = pathway2embedding[pathway]
        if embedding_matrix is None:
            embedding_matrix = embedding_array
        else:
            embedding_matrix = np.vstack((embedding_matrix, embedding_array))
    return embedding_matrix, pathway_list


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors.
    
    Args:
        vec_a, vec_b: numpy arrays or lists of numbers
    
    Returns:
        float: cosine similarity, between -1 and 1
    """
    dot_product = np.dot(vec_a, vec_b)
    mag_a = np.linalg.norm(vec_a)
    mag_b = np.linalg.norm(vec_b)
    return dot_product / (mag_a * mag_b) 