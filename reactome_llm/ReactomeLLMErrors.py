class NoAbstractFoundError(Exception):
    def __init__(self, gene):
        self.message = 'Nothing is found in PubMed for {} about interactions, reactions and pathways'.format(gene)
        super().__init__(self.message)


class NoInteractingPathwayFoundError(Exception):
    def __init__(self, gene):
        self.message = 'No interacting pathway is found for {}. Try to reduce the cutoff value for functional interactions.'.format(gene)
        super().__init__(self.message)


class NoAbstractSupportingInteractingPathwayError(Exception):
    def __init__(self, gene):
        self.message = 'Cannot find any abstract to support the predicted interacting pathway for {}.'.format(gene)
        super().__init__(self.message)


class NoAbstractSupportingProteinInteractions(Exception):
    def __init__(self, gene):
        self.message = 'Cannot find any abstract to support the protein interactions for {}.'.format(gene)
        super().__init__(self.message)


class PubMedFullTextPDFNotFoundError(Exception):
    def __init__(self, pmid):
        self.message = 'Cannot find a URL to download the full text PDF for {}'.format(pmid)
        super().__init__(self.message)




