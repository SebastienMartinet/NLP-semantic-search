import streamlit as st
from search import load_abstracts, TfidfSearch, SemanticSearch, HybridSearch,BERTSearch
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(
    page_title="ADS Abstract Search",
    layout="wide",  # full-width layout
    initial_sidebar_state="expanded"
)

from PIL import Image
# Load image
# Display in sidebar
st.sidebar.image("assets/avatar_open_to_work.png", caption="Sébastien MARTINET", use_container_width=True)

# -----------------------------
# Load dataset
# -----------------------------
# st.title("Astrophysics Abstract Search")
st.title("NLP Semantic Search")
st.subheader("Astrophysics abstracts")
data, texts = load_abstracts()

# -----------------------------
# Sidebar: select retrieval method
# -----------------------------
methods = ["TF-IDF", "Semantic", "Hybrid", "BERT"]

method = st.sidebar.selectbox(
    "Retrieval method",
    methods,
    index=methods.index("Hybrid")
)

top_k = st.sidebar.slider("Number of results", min_value=1, max_value=200, value=25)
alpha = st.sidebar.slider("Hybrid α (semantic weight)", 0.0, 1.0, 0.8, 0.05)

# -----------------------------
# Caching models
# -----------------------------
@st.cache_resource
def init_tfidf():
    return TfidfSearch(texts)

@st.cache_resource
def init_semantic():
    return SemanticSearch(texts, cached_embeddings_path="data/embeddings.npy")

@st.cache_resource
def init_bert():
    model = SentenceTransformer("allenai-specter")
    try:
        embeddings = np.load("data/bert_embeddings.npy")
    except FileNotFoundError:
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        np.save("data/bert_embeddings.npy", embeddings)
    return model, embeddings

tfidf_model = init_tfidf()
semantic_model = init_semantic()
hybrid_model = HybridSearch(semantic_model, data, alpha=alpha)
model, bert_embeddings = init_bert()
bert_model = BERTSearch(model, bert_embeddings)

# # -----------------------------
# # BERT Search class
# # -----------------------------
# class BERTSearch:
#     def __init__(self, model, embeddings):
#         self.model = model
#         self.embeddings = embeddings

#     def search(self, query, top_k=10):
#         query_emb = self.model.encode([query], convert_to_numpy=True)
#         sims = cosine_similarity(query_emb, self.embeddings)[0]
#         idx = np.argsort(-sims)[:top_k]
#         scores = sims[idx]
#         return idx, scores

# bert_model = BERTSearch(bert_model, bert_embeddings)

# -----------------------------
# Utility: Build node graph with clustering
# -----------------------------
from sklearn.cluster import AgglomerativeClustering
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer

def cluster_keywords(texts, top_n=3, exclude_words=None):
    if not texts:
        return []

    if exclude_words is None:
        exclude_words = {
            "sub", "sup", "frac", "cdot",
            "sub sub", "sup sup", "gt", "lt",
            "mathrm", "mathit", "alpha", "beta","pdf","http","href",
            "loc", "loc post", "post"
        }

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )

    X = vectorizer.fit_transform(texts)
    counts = X.toarray().sum(axis=0)
    words = vectorizer.get_feature_names_out()

    filtered = [
        (w, c) for w, c in zip(words, counts)
        if w not in exclude_words and not any(x in w for x in exclude_words)
    ]

    filtered.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in filtered[:top_n]]

def build_similarity_graph(
    embeddings,
    top_results,
    threshold=0.5,
    cluster_distance_threshold=0.4,
    min_cluster_size=3,
    highlight_cluster=None
):
    embeddings = np.asarray(embeddings)
    n_total = len(embeddings)

    if n_total == 0:
        return None, None, {}, {}, {}

    min_cluster_size = min(min_cluster_size, n_total)
    
    if n_total < 2:
        labels = np.zeros(n_total, dtype=int)
        cluster_dict = {0: [0]}
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=cluster_distance_threshold
        )
        raw_labels = clustering.fit_predict(embeddings)
        counts = Counter(raw_labels)

        labels = np.zeros(n_total, dtype=int)
        cluster_dict = defaultdict(list)
        next_label = 0
        singleton_indices = []

        for lbl in sorted(set(raw_labels)):
            indices = [i for i, x in enumerate(raw_labels) if x == lbl]
            if len(indices) >= min_cluster_size:
                for i in indices:
                    labels[i] = next_label
                    cluster_dict[next_label].append(i)
                next_label += 1
            else:
                singleton_indices.extend(indices)

        if singleton_indices:
            for i in singleton_indices:
                labels[i] = next_label
                cluster_dict[next_label].append(i)
            next_label += 1

    palette = plt.get_cmap("tab10").colors
    cluster_colors = {}
    for lbl in cluster_dict:
        if lbl == next_label - 1 and singleton_indices:
            cluster_colors[lbl] = "rgba(150,150,150,0.4)"
        else:
            cluster_colors[lbl] = "#{:02x}{:02x}{:02x}".format(
                int(palette[lbl % len(palette)][0] * 255),
                int(palette[lbl % len(palette)][1] * 255),
                int(palette[lbl % len(palette)][2] * 255),
            )

    cluster_keywords_dict = {}
    for lbl, indices in cluster_dict.items():
        texts = [
            top_results[i].get("abstract", "")
            for i in indices
            if top_results[i].get("abstract")
        ]
        cluster_keywords_dict[lbl] = cluster_keywords(texts)

    G = nx.Graph()
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = float(
                np.dot(embeddings[i], embeddings[j]) /
                (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            )
            if sim <= threshold:
                continue
            is_edge_highlight = (
                highlight_cluster is not None and
                labels[i] == highlight_cluster and
                labels[j] == highlight_cluster
            )
            G.add_edge(
                i, j,
                weight=sim,
                title=f"Sim={sim:.2f}",
                width=6 if is_edge_highlight else 1,
                color=cluster_colors[labels[i]] if is_edge_highlight else "#999999"
            )

    for i, paper in enumerate(top_results):
        lbl = labels[i]
        title = paper.get("title", "")
        short = title[:40] + "…" if len(title) > 40 else title
        is_node_highlight = (
            highlight_cluster is not None and lbl == highlight_cluster
        )
        fa = first_author(paper)
        year = paper.get("year", "n.d.")
        tooltip = (
            f"{title}\n"
            f"{fa}+{year}\n"
            f"Citations: {paper.get('citations',0)}\n"
            f"Cluster: {', '.join(cluster_keywords_dict[lbl])}"
        )
        G.add_node(
            i,
            label=short,
            title=tooltip,
            value=45 if is_node_highlight else 15,
            color=cluster_colors[lbl]
        )

    net = Network(
        height="600px",
        width="100%",
        bgcolor="rgba(0,0,0,0)",
        directed=False,
        notebook=False
    )
    net.from_nx(G)
    for edge in net.edges:
        i = edge["from"]
        j = edge["to"]
        if (
            highlight_cluster is not None
            and labels[i] == highlight_cluster
            and labels[j] == highlight_cluster
        ):
            edge["width"] = 6
            edge["color"] = cluster_colors[labels[i]]
        else:
            edge["width"] = 3
            edge["color"] = "#cccccc"
    net.force_atlas_2based(
        gravity=-30,
        central_gravity=0.01,
        spring_length=140,
        spring_strength=0.08,
        damping=0.4
    )
    return net, labels, cluster_dict, cluster_keywords_dict, cluster_colors

def first_author(paper):
    if "authors" in paper and paper["authors"]:
        return paper["authors"][0]
    if "author" in paper and paper["author"]:
        return paper["author"][0]
    return "Unknown"

def ads_link(d):
    link_pdf=f"https://ui.adsabs.harvard.edu/link_gateway/{d.get('bibcode')}/ESOURCE"
    link_ads_page=f"https://ui.adsabs.harvard.edu/abs/{d.get('bibcode')}/abstract"
    return link_pdf,link_ads_page

#### Comparison tools
def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def cluster_stats(cluster_dict):
    if not cluster_dict:
        return {
            "n_clusters": 0,
            "largest_cluster": 0,
            "singleton_fraction": 0.0
        }

    sizes = [len(v) for v in cluster_dict.values()]
    n_total = sum(sizes)

    return {
        "n_clusters": len(sizes),
        "largest_cluster": max(sizes),
        "singleton_fraction": sum(1 for s in sizes if s == 1) / len(sizes)
    }

def get_clusters_for(method_name, idxs):
    if method_name == "TF-IDF":
        return None

    if method_name == "Hybrid":
        emb = hybrid_model.semantic.embeddings[idxs]
    elif method_name == "Semantic":
        emb = semantic_model.embeddings[idxs]
    else:  # BERT
        emb = bert_model.embeddings[idxs]

    net, _, cluster_dict, _, _ = build_similarity_graph(
        emb,
        [data[i] for i in idxs],
        threshold=0.7
    )
    return cluster_dict

def arrow_safe(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def explain_tfidf(tfidf_model, query, doc_index, top_n=5):
    vec = tfidf_model.vectorizer
    q_vec = vec.transform([query])
    d_vec = tfidf_model.tfidf[doc_index]

    contrib = q_vec.multiply(d_vec).toarray()[0]
    terms = vec.get_feature_names_out()

    top = sorted(
        [(terms[i], contrib[i]) for i in contrib.nonzero()[0]],
        key=lambda x: x[1],
        reverse=True
    )

    return top[:top_n]

def explain_semantic(model, query, abstract, top_n=2):
    if not abstract:
        return []

    sentences = split_sentences(abstract)
    sent_emb = model.encode(sentences, convert_to_numpy=True)
    q_emb = model.encode([query], convert_to_numpy=True)

    sims = cosine_similarity(q_emb, sent_emb)[0]
    top_idx = np.argsort(-sims)[:top_n]

    return [(sentences[i], sims[i]) for i in top_idx]

def explain_hybrid(hybrid_model, idx):
    return {
        "semantic": float(hybrid_model.last_semantic_scores[idx]),
        "citations": float(hybrid_model.last_citation_scores[idx]),
        "hybrid": float(hybrid_model.last_hybrid_scores[idx]),
    }

# -----------------------------
# Method description
# -----------------------------
if method == "TF-IDF":
    st.info(
        "TF-IDF Search: ranks papers by exact keyword overlap. "
        "Fast and interpretable, but limited to lexical matching."
    )

elif method == "Semantic":
    st.info(
        "Semantic Search: uses sentence embeddings to retrieve papers "
        "based on semantic similarity, even without shared keywords."
    )

elif method == "Hybrid":
    st.info(
        "Hybrid Search: combines semantic similarity with citation impact. "
        "Balances topical relevance and scientific influence."
    )

elif method == "BERT":
    st.info(
        "BERT Search: transformer-based embeddings capture fine semantic meaning. "
        "Results are often very coherent, sometimes forming a single cluster."
    )

# -----------------------------
# User input
# -----------------------------
query = st.text_input("Enter your query:",placeholder="e.g. Machine Learning")

if query:
    if method == "TF-IDF":
        idx, scores = tfidf_model.search(query, top_k=top_k)
    elif method == "Semantic":
        idx, scores = semantic_model.search(query, top_k=top_k)
    elif method == "Hybrid":
        hybrid_model.alpha = alpha
        idx, scores = hybrid_model.search(query, top_k=top_k)
    else:  # BERT
        idx, scores = bert_model.search(query, top_k=top_k)

    if isinstance(idx, int):
        idx = [idx]
    top_results = [data[i] for i in idx]

    highlight_lbl = st.session_state.get("highlight_cluster", None)

    results_by_method = {}

    # --- TF-IDF ---
    idx_tfidf, _ = tfidf_model.search(query, top_k=top_k)
    results_by_method["TF-IDF"] = idx_tfidf if isinstance(idx_tfidf, list) else [idx_tfidf]

    # --- Semantic ---
    idx_sem, _ = semantic_model.search(query, top_k=top_k)
    results_by_method["Semantic"] = idx_sem

    # --- Hybrid ---
    hybrid_model.alpha = alpha
    idx_hybrid, _ = hybrid_model.search(query, top_k=top_k)
    results_by_method["Hybrid"] = idx_hybrid

    # --- BERT ---
    idx_bert, _ = bert_model.search(query, top_k=top_k)
    results_by_method["BERT"] = idx_bert

    comparison = {}

    for method_name, idxs in results_by_method.items():
        # Flatten and convert to Python int list
        idxs = np.array(idxs).flatten().astype(int).tolist()
        
        citations = [data[i].get("citations", 0) for i in idxs]
        
        cluster_dict = get_clusters_for(method_name, idxs)
        
        stats = cluster_stats(cluster_dict) if cluster_dict else {
            "n_clusters": 0,
            "largest_cluster": 0,
            "singleton_fraction": 0.0
        }
        
        comparison[method_name] = {
            **stats,
            "mean_citations": float(np.mean(citations)),
            "median_citations": float(np.median(citations)),
        }


    for m in results_by_method:
        # Flatten, convert to int, and ensure plain Python list
        results_by_method[m] = np.array(results_by_method[m]).flatten().astype(int).tolist()

    methods = list(results_by_method.keys())
    overlap = {}

    for m1, m2 in combinations(methods, 2):
        overlap[f"{m1} ↔ {m2}"] = jaccard(
                results_by_method[m1],
                results_by_method[m2]
            )


    if method in ["Semantic", "Hybrid", "BERT"]:
        st.write("### Embedding Similarity Graph (top-k abstracts)")
        if "highlight_cluster" not in st.session_state:
            st.session_state["highlight_cluster"] = None

        top_embeddings = (
            hybrid_model.semantic.embeddings[idx]
            if method == "Hybrid"
            else (semantic_model.embeddings[idx] if method=="Semantic" else bert_embeddings[idx])
        )
        top_results = [data[i] for i in idx]
        with st.spinner("Rendering node graph..."):
            net, labels, cluster_dict, cluster_keywords, cluster_colors = build_similarity_graph(
                top_embeddings,
                top_results,
                threshold=0.7,
                highlight_cluster=st.session_state["highlight_cluster"]
            )
            if net is not None:
                if st.sidebar.button("Clear highlight"):
                    st.session_state["highlight_cluster"] = None
                    st.rerun()

                sorted_clusters = sorted(
                    cluster_dict.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )
                for lbl in sorted(cluster_dict.keys()):
                    indices = cluster_dict[lbl]
                    kw = ", ".join(cluster_keywords[lbl])
                    if st.sidebar.button(f"Cluster {lbl}: {kw} ({len(indices)})"):
                        st.session_state["highlight_cluster"] = lbl
                        st.rerun()

                st.markdown(
                    """
                    <style>
                    iframe {
                        border: none !important;
                        background: transparent !important;
                    }
                    .stHtmlFrame {
                        background: transparent !important;
                        border: none !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                net.write_html("graph.html")
                with open("graph.html", "r", encoding="utf-8") as f:
                    html = f.read()
                injection = """
                <style>
                    body { margin: 0 !important; background: transparent !important; }
                    canvas { background: transparent !important; }
                    iframe { border: none !important; background: transparent !important; }
                </style>
                """
                html = html.replace("</head>", injection + "\n</head>")
                cluster_tag = st.session_state.get("highlight_cluster", "none")
                html = f"<!-- cluster={cluster_tag} -->\n" + html
                components.html(html, height=650, scrolling=False)
                st.markdown("""
                <style>
                iframe {
                    border: none !important;
                    background: transparent !important;
                }
                .stHtmlFrame {
                    border: none !important;
                    background: transparent !important;
                }
                </style>
                """, unsafe_allow_html=True)
            else:
                st.write("Not enough abstracts to build similarity graph.")

        highlight_lbl = st.session_state["highlight_cluster"]
        if highlight_lbl is not None:
            st.subheader(f"Papers in Cluster {highlight_lbl}")
            for i in cluster_dict[highlight_lbl]:
                d=top_results[i]
                fa = first_author(d)
                st.markdown(
                    f"*{fa} et al., {d['year']} — {d['title']} — {d['journal']} "
                    f"(citations: {d.get('citations', 0)})*"
                )

    with st.expander("🔍 Comparison of retrieval methods",expanded=False):
        st.caption(
            "Methods are compared using proxy metrics capturing semantic coherence, "
            "citation bias, and result diversity (no labeled relevance data)."
        )
        # --- Table ---
        df = pd.DataFrame.from_dict(comparison, orient='index')
        fig, axes = plt.subplots(1, 3, figsize=(12, 3),dpi=300)

        # sns.barplot(x=df.index, y=df["mean_citations"], ax=axes[0])
        # axes[0].set_title("Mean Citations")
        x = np.arange(len(df.index))  # label locations
        width = 0.35  # width of the bars
        rects1 = axes[0].bar(x - width/2, df["mean_citations"], width, label='Mean')
        rects2 = axes[0].bar(x + width/2, df["median_citations"], width, label='Median', color='salmon')
        axes[0].set_ylabel("Citations")
        axes[0].set_title("Mean vs Median Citations by Retrieval Method")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df.index)
        axes[0].legend()
        sns.barplot(x=df.index, y=df["n_clusters"], ax=axes[1])
        axes[1].set_title("Number of Clusters")

        sns.barplot(x=df.index, y=df["largest_cluster"], ax=axes[2])
        axes[2].set_title("Largest Cluster Size")

        # sns.barplot(x=df.index, y=df["singleton_fraction"], ax=axes[1, 1])
        # axes[1, 1].set_title("Singleton Fraction")

        fig.tight_layout()
        st.pyplot(fig,clear_figure=True)        
        # --- Table ---
        st.write("### Summary metrics")
        table = {
            m: {
                "Mean citations": f"{comparison[m]['mean_citations']:.1f}",
                "Median citations": f"{comparison[m]['median_citations']:.1f}",
                "Clusters": comparison[m]["n_clusters"],
                "Largest cluster": comparison[m]["largest_cluster"],
                "Singleton fraction": f"{comparison[m]['singleton_fraction']:.2f}",
            }
            for m in comparison
        }
        df_table = pd.DataFrame.from_dict(table, orient="index")
        st.dataframe(arrow_safe(df_table))

        # --- Overlap ---
        st.write("### Rank overlap (Jaccard index)")
        st.caption(
    "The Jaccard index measures the overlap between result sets of two retrieval methods. "
    "0 = no overlap, 1 = identical results.")
        for k, v in overlap.items():
            st.write(f"- **{k}**: {v:.2f}")
    st.write(f"### Top {top_k} results ({method} search)")
    for rank, i in enumerate(idx, start=1):
        d = data[i]

        st.markdown(f"**{rank}. {d['title']}**")
        st.caption(f"{first_author(d)} et al., {d['year']} (Citations: {d['citations']})")
        ads_pdf, ads_page = ads_link(d)
        st.markdown(
    f"[View ADS]({ads_page}) &nbsp;|&nbsp; [View PDF]({ads_pdf})",
    unsafe_allow_html=True
)
        st.write(d["abstract"][:500] + "…")
        if rank ==1:
            expanded_status=True
        else:
            expanded_status=False
        with st.expander("ⓘ",expanded=expanded_status):
#             st.markdown(
#     """F
#     <style>
#     /* Expander header text */
#     div[data-testid="stExpander"] > details > summary {
#         font-size: 0.85rem;
#         opacity: 0.75;
#     }

#     /* Arrow icon */
#     div[data-testid="stExpander"] summary svg {
#         transform: scale(0.8);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
            if method == "TF-IDF":
                terms = explain_tfidf(tfidf_model, query, i)
                st.write("Top matching terms:")
                for t, w in terms:
                    st.write(f"- **{t}** ({w:.3f})")

            elif method in ["Semantic", "BERT"]:
                expl = explain_semantic(
                    model if method=="BERT" else semantic_model.model,
                    query,
                    d["abstract"]
                )
                for sent, score in expl:
                    st.write(f"• *{sent}*  \n  Similarity: `{score:.2f}`")
            elif method == "Hybrid":
                scores = explain_hybrid(hybrid_model, i)
                st.progress(alpha*scores["semantic"], text="Semantic relevance")
                st.progress((1-alpha)*scores["citations"], text="Citation influence")
                st.write(
                    f"Semantic: `{alpha*scores['semantic']:.2f}` | "
                    f"Citations: `{(1-alpha)*scores['citations']:2f}` | "
                    f"Final: `{scores['hybrid']:.2f}`"
                )
                st.write("**Semantic relevance**")
                expl = explain_semantic(
                    semantic_model.model,
                    query,
                    d["abstract"]
                )
                for sent, score in expl:
                    st.write(f"• *{sent}*  \n  Similarity: `{score:.2f}`")

        st.markdown("---")
    # if method == "Hybrid":
    #     sem_scores, citation_scores = [], []
    #     for i in idx:
    #         sem_scores.append(hybrid_model.semantic.embeddings[i])
    #         citation_scores.append(hybrid_model.citations[i])
    #     sem_norm = np.array([np.linalg.norm(e) for e in sem_scores])
    #     citation_norm = np.array(citation_scores[:top_k])
    #     hybrid_scores = alpha*sem_norm + (1-alpha)*citation_norm
    #     st.write("### Hybrid Score Decomposition")
    #     fig, ax = plt.subplots(figsize=(8,4))
    #     ax.bar(range(top_k), sem_norm, label="Semantic")
    #     ax.bar(range(top_k), (1-alpha)*citation_norm, bottom=alpha*sem_norm, label="Citations")
    #     ax.set_xticks(range(top_k))
    #     ax.set_xticklabels([r["title"][:20]+"..." for r in top_results], rotation=45, ha='right')
    #     ax.set_ylabel("Hybrid score")
    #     ax.legend()
    #     st.pyplot(fig)

