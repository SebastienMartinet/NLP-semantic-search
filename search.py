import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Data loading
# -----------------------------

def load_abstracts(path="data/abstracts.json"):
    with open(path, "r") as f:
        data = json.load(f)

    texts = [d["abstract"] for d in data]
    return data, texts


# -----------------------------
# TF-IDF search
# -----------------------------

class TfidfSearch:
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20_000
        )
        self.tfidf = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=10):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf)[0]
        idx = np.argsort(scores)[::-1][:top_k]
        return idx, scores



# -----------------------------
# Semantic search (transformers)
# -----------------------------

class SemanticSearch:
    def __init__(self, texts, model_name="all-MiniLM-L6-v2", cached_embeddings_path=None):
        from sentence_transformers import SentenceTransformer, util
        self.model = SentenceTransformer(model_name,cache_folder="models")
        
        if cached_embeddings_path:
            import os
            import numpy as np
            if os.path.exists(cached_embeddings_path):
                self.embeddings = np.load(cached_embeddings_path)
                print(f"Loaded cached embeddings from {cached_embeddings_path}")
            else:
                self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
                np.save(cached_embeddings_path, self.embeddings)
                print(f"Saved embeddings to {cached_embeddings_path}")
        else:
            self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def search(self, query, top_k=10):
        """
        Return indices and scores of top_k most similar embeddings
        """
        from sentence_transformers import util
        q_emb = self.model.encode(query, convert_to_numpy=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0].numpy()
        idx = np.argsort(scores)[::-1][:top_k]
        return idx, scores


    def similarity(self, query):
        """
        Returns cosine similarity between query and ALL documents
        """
        q_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        return sims
# -----------------------------
# Hybrid search (semantic + citations)
# -----------------------------

def normalize(values):
    v = np.array(values, dtype=float)
    return (v - v.min()) / (v.max() - v.min() + 1e-8)


class HybridSearch:
    def __init__(self, semantic_model, data, alpha=0.8):
        self.semantic = semantic_model
        self.data = data
        self.alpha = alpha
        self.citations = np.array([d.get("citations", 0) for d in data])

        # explainability buffers
        self.last_semantic_scores = None
        self.last_citation_scores = None
        self.last_hybrid_scores = None


    def search(self, query, top_k=10):
        # semantic similarity
        sem_scores = self.semantic.similarity(query)

        # semantic normalization (NumPy 2.0 safe)
        sem_range = np.ptp(sem_scores)
        sem_norm = (sem_scores - sem_scores.min()) / (sem_range + 1e-8)

        # citation normalization (log-scaled)
        cit_log = np.log1p(self.citations)
        cit_range = np.ptp(cit_log)
        cit_norm = (cit_log - cit_log.min()) / (cit_range + 1e-8)

        # hybrid score
        hybrid_scores = self.alpha * sem_norm + (1 - self.alpha) * cit_norm

        # ranking
        idx = np.argsort(-hybrid_scores)[:top_k]

        # explainability buffers
        self.last_semantic_scores = sem_norm
        self.last_citation_scores = cit_norm
        self.last_hybrid_scores = hybrid_scores

        return idx.tolist(), hybrid_scores[idx]



# -----------------------------
# BERT search 
# -----------------------------


class BERTSearch:
    def __init__(self, model, embeddings):
        self.model = model
        self.embeddings = embeddings

    def search(self, query, top_k=10):
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        return idx.tolist(), sims[idx].tolist()
