from sentence_transformers import SentenceTransformer
from .cosine_similarity import cosineSim
from .base import embMetrics

class sentenceTransformer(embMetrics):

    def __init__(self) -> None:
        self.model_text = SentenceTransformer("all-MiniLM-L6-v2")
        
    def compute(self, text_1, text_2):
        # Our sentences to encode
        # Sentences are encoded by calling model.encode()
        res = 0
        if text_1 and text_2:
            embeddingS = self.compute_textEmb(text_1)
            embeddingGT = self.compute_textEmb(text_2)
                # Print the embeddings
            res = cosineSim(embeddingS, embeddingGT)
        return res