# -------------------------------------------------
# 0️⃣ IMPORTS
# -------------------------------------------------
import nltk
import numpy as np
from datasets import Dataset

from nltk.tokenize import word_tokenize, sent_tokenize
from textstat import flesch_reading_ease

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama

nltk.download("punkt")


# -------------------------------------------------
# 1️⃣ INPUT DOCUMENT
# -------------------------------------------------
markdown = """
Deep learning is a subset of machine learning that uses neural networks with many layers.
It automatically learns hierarchical features like edges → shapes → objects.
It is used in NLP, computer vision, speech recognition, and generative AI.
"""


# -------------------------------------------------
# 2️⃣ CHUNKING
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.create_documents([markdown])

for i, ch in enumerate(chunks):
    ch.metadata["id"] = str(i)


# -------------------------------------------------
# 3️⃣ VECTOR STORE
# -------------------------------------------------
emb = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_store = FAISS.from_documents(chunks, emb)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# -------------------------------------------------
# 4️⃣ LLM (Ollama)
# -------------------------------------------------
llm = ChatOllama(model="mistral:latest", temperature=0.2)


# -------------------------------------------------
# 5️⃣ PROMPT TEMPLATE
# -------------------------------------------------
prompt = PromptTemplate(
    template="""
Answer only using the context. If missing, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


# -------------------------------------------------
# 6️⃣ QUESTIONS & REFERENCES
# -------------------------------------------------
questions = [
    "What is deep learning?",
    "How does deep learning learn hierarchical features?"
]

references = [
    "Deep learning uses neural networks with many layers.",
    "Deep learning learns hierarchical features through layered neural networks."
]


# ================================================================
# 🔥 RAG PIPELINE: RETRIEVE + GENERATE
# ================================================================
answers = []
retrieved_docs_all = []

for q in questions:
    docs = retriever.invoke(q)
    retrieved_docs = [d.page_content for d in docs]
    retrieved_docs_all.append(retrieved_docs)

    context_text = "\n".join(retrieved_docs)
    full_prompt = prompt.invoke({"context": context_text, "question": q})

    response = llm.invoke(full_prompt).content
    answers.append(response)


# ===================================================================
# 🔥 METRIC FUNCTIONS (YOUR EXACT FUNCTIONS — UNMODIFIED)
# ===================================================================

# 1. Answer Relevance & Context Utilization
def answer_relevance_context_utilization(responses, references, retrieved_docs, top_k=5):
    relevance_scores = []
    context_scores = []

    for resp, ref, docs in zip(responses, references, retrieved_docs):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        ref_words = set(nltk.word_tokenize(ref.lower()))

        relevance_scores.append(
            len(resp_words & ref_words) / len(ref_words) if ref_words else 0
        )

        doc_words = set(w for d in docs[:top_k] for w in nltk.word_tokenize(d.lower()))
        context_scores.append(
            len(resp_words & doc_words) / len(resp_words) if resp_words else 0
        )

    return {
        "Answer Relevance": np.mean(relevance_scores),
        "Context Utilization": np.mean(context_scores)
    }


# 2. Groundedness
def groundedness(responses, retrieved_docs, top_k=3):
    scores = []
    for resp, docs in zip(responses, retrieved_docs):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        doc_words = set(w for d in docs[:top_k] for w in nltk.word_tokenize(d.lower()))
        overlap = len(resp_words & doc_words) / len(resp_words) if resp_words else 0
        scores.append(overlap)
    return np.mean(scores)


# 3. Hallucination Rate
def hallucination_rate(responses, retrieved_docs, top_k=3):
    rates = []
    for resp, docs in zip(responses, retrieved_docs):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        doc_words = set(w for d in docs[:top_k] for w in nltk.word_tokenize(d.lower()))
        unsupported = len(resp_words - doc_words)
        hall = unsupported / len(resp_words) if resp_words else 0
        rates.append(hall)
    return np.mean(rates)


# 4. Coherence + Readability
def response_coherence_readability(responses):
    coherence_scores = []
    readability_scores = []

    for resp in responses:
        sentences = nltk.sent_tokenize(resp)
        words = nltk.word_tokenize(resp)

        coherence = len(words) / len(sentences) if sentences else 0
        coherence_scores.append(coherence)

        readability = flesch_reading_ease(resp)
        readability_scores.append(readability)

    return {
        "Average Coherence (words/sentence)": np.mean(coherence_scores),
        "Average Readability (Flesch)": np.mean(readability_scores)
    }


# 5. Query Relevancy
def relevancy_score(responses, queries):
    scores = []
    for resp, query in zip(responses, queries):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        query_words = set(nltk.word_tokenize(query.lower()))
        overlap = len(resp_words & query_words)
        relevancy = overlap / len(query_words) if query_words else 0
        scores.append(relevancy)
    return np.mean(scores)


# ================================================================
# 🔥 METRICS PER QUESTION (Option A)
# ================================================================
print("\n==================== RAG METRICS (PER QUESTION) ====================\n")

for i in range(len(questions)):
    q = questions[i]
    a = answers[i]
    ref = references[i]
    docs = retrieved_docs_all[i]

    # Calculate per-question values
    rel = answer_relevance_context_utilization([a], [ref], [docs])
    grd = groundedness([a], [docs])
    hal = hallucination_rate([a], [docs])
    coh = response_coherence_readability([a])
    q_rel = relevancy_score([a], [q])

    # Print block
    print(f"\n-------------------- QUESTION {i+1} --------------------")
    print("Question:", q)
    print("Answer:", a)
    print("\nAnswer Relevance:", round(rel["Answer Relevance"], 4))
    print("Context Utilization:", round(rel["Context Utilization"], 4))
    print("Groundedness:", round(grd, 4))
    print("Hallucination Rate:", round(hal, 4))
    print("Coherence:", round(coh["Average Coherence (words/sentence)"], 4))
    print("Readability (Flesch):", round(coh["Average Readability (Flesch)"], 4))
    print("Query Relevancy:", round(q_rel, 4))

print("\n=================================================================\n")
