# ===========================================
# ⭐ COMPLETE RAG PIPELINE + RETRIEVAL METRICS + RAGAS
# ===========================================

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import math
load_dotenv()

# --------------------------
# 1. INPUT DOCUMENT
# --------------------------
markdown = '''Deep learning is a powerful subset of machine learning within AI that uses multi-layered artificial neural networks, inspired by the human brain, to automatically learn complex patterns and features from vast amounts of data, enabling advanced tasks like image/speech recognition, natural language processing (NLP), and powering modern AI like large language models (LLMs) and self-driving cars. The "deep" in its name refers to the many layers of interconnected nodes (neurons) that process information hierarchically, extracting increasingly abstract representations from raw input.  
How it Works
Neural Networks: It uses networks with input, hidden (multiple layers), and output layers, mimicking brain functions. 
Layered Learning: Each layer learns simpler features from the layer below, building up to complex concepts (e.g., edges -> shapes -> objects in images). 
Automatic Feature Extraction: Unlike traditional ML, deep learning automatically discovers relevant features from data, eliminating manual engineering. 
Key Applications
Computer Vision: Image recognition, object detection (e.g., in self-driving cars). 
Natural Language Processing (NLP): Understanding and generating human language (e.g., ChatGPT, translation). 
Speech Recognition: Transcribing voice to text. 
Generative AI: Creating new images, text, and music (e.g., Midjourney, DALL-E). 
Why it's Important
Drives most state-of-the-art AI advancements.
Handles large, unstructured datasets effectively.
Enables intelligent automation of complex tasks.'''

# --------------------------
# 2. CHUNK THE DOCUMENT
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=10)
chunks = splitter.create_documents([markdown])

# Add metadata IDs for retrieval metrics
for i, chunk in enumerate(chunks):
    chunk.metadata["id"] = str(i)

print("Chunks created:", len(chunks))

# --------------------------
# 3. EMBEDDINGS + FAISS
# --------------------------
embeddings2 = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings2)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print("FAISS vector store ready!")

# --------------------------
# 4. LLM
# --------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

print("LLM loaded!")

# --------------------------
# 5. PROMPT TEMPLATE
# --------------------------
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided context.
    If the context is insufficient, say "I don't know."

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

print("Prompt template ready!")

# --------------------------
# 6. GENERATE ANSWER
# --------------------------
question = "What is deep learning?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = llm.invoke(final_prompt).content

print("\n🟢 FINAL ANSWER →", answer)

# ===========================================
# ⭐ RETRIEVAL METRICS
# ===========================================

def recall_at_k(retrieved, relevant_docs, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc.metadata["id"] in relevant_docs)
    return hits / len(relevant_docs)

def precision_at_k(retrieved, relevant_docs, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc.metadata["id"] in relevant_docs)
    return hits / k

def mrr(retrieved, relevant_docs):
    for rank, doc in enumerate(retrieved, start=1):
        if doc.metadata["id"] in relevant_docs:
            return 1 / rank
    return 0

def hit_rate_at_k(retrieved, relevant_docs, k):
    retrieved_k = retrieved[:k]
    return int(any(doc.metadata["id"] in relevant_docs for doc in retrieved_k))

def dcg(docs, rel_scores):
    score = 0
    for i, doc in enumerate(docs, start=1):
        rel = rel_scores.get(doc.metadata["id"], 0)
        score += rel / math.log2(i + 1)
    return score

def ndcg(retrieved, relevant_docs, rel_scores):
    # Convert relevant doc IDs → doc objects
    ideal_docs = []
    for doc_id in sorted(relevant_docs, key=lambda d: rel_scores.get(d, 0), reverse=True):
        for doc in retrieved:
            if doc.metadata["id"] == doc_id:
                ideal_docs.append(doc)
                break

    # Avoid divide by zero
    if len(ideal_docs) == 0:
        return 0

    ideal_dcg = dcg(ideal_docs, rel_scores)
    if ideal_dcg == 0:
        return 0

    return dcg(retrieved, rel_scores) / ideal_dcg

# --------------------------
# AUTO-DETECT RELEVANT DOCS
# --------------------------
keyword = "deep learning".lower()

relevant_docs = {
    doc.metadata["id"]
    for doc in chunks
    if keyword in doc.page_content.lower()
}

relevance_scores = {doc_id: 1 for doc_id in relevant_docs}

print("\nAuto-selected relevant docs:", relevant_docs)

# --------------------------
# PRINT RETRIEVAL METRICS
# --------------------------
print("\n===================================")
print("📌 RETRIEVAL METRICS")
print("===================================")
print("Recall@5      :", recall_at_k(retrieved_docs, relevant_docs, 5))
print("Precision@5   :", precision_at_k(retrieved_docs, relevant_docs, 5))
print("MRR           :", mrr(retrieved_docs, relevant_docs))
print("Hit Rate@5    :", hit_rate_at_k(retrieved_docs, relevant_docs, 5))
print("nDCG          :", ndcg(retrieved_docs, relevant_docs, relevance_scores))

# ===========================================
# ⭐ RAGAS EVALUATION
# ===========================================
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness
from datasets import Dataset

ragas_data = {
    "question": [question],
    "answer": [answer],
    "contexts": [[doc.page_content for doc in retrieved_docs]],
    "ground_truths": ["Deep learning refers to neural networks with many layers."]
}

dataset = Dataset.from_dict(ragas_data)

ragas_result = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall, answer_relevancy, faithfulness]
)

print("\n===================================")
print("📌 RAGAS METRICS")
print("===================================")
print(ragas_result.to_pandas())
