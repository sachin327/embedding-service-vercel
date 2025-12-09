from embedding import Embedding

EMBEDDING_SERVICE = None


def get_embedding_service():
    global EMBEDDING_SERVICE
    if EMBEDDING_SERVICE is None:
        EMBEDDING_SERVICE = Embedding()
    return EMBEDDING_SERVICE
