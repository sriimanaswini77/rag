# ===========================================
# ⭐ RAG PIPELINE + RAGAS METRICS (Python 3.10+)
# ===========================================

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings, 
    ChatGoogleGenerativeAI
)
from ragas import evaluate
from ragas.metrics import (
    context_precision, 
    context_recall, 
    answer_relevancy, 
    faithfulness
)
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()


# --------------------------
# 1. INPUT DOCUMENT
# --------------------------
markdown = '''Deep learning is a powerful subset of machine learning within AI that uses multi-layered artificial neural networks, inspired by the human brain, ...'''


# --------------------------
# 2. CHUNKING
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
chunks = splitter.create_documents([markdown])


# --------------------------
# 3. ADD METADATA IDs
# --------------------------
for i, chunk in enumerate(chunks):
    chunk.metadata["id"] = str(i)


# --------------------------
# 4. FAISS VECTOR STORE
# --------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# --------------------------
# 5. LLM
# --------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


# --------------------------
# 6. PROMPT TEMPLATE
# --------------------------
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided context.
    If the answer isn't in the context, say "I don't know."

    Context:
    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)


# --------------------------
# 7. RUN THE RAG PIPELINE
# --------------------------
question = "What is deep learning?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})

answer = llm.invoke(final_prompt).content

print("\n🟢 FINAL ANSWER →", answer)


# ===========================================
# ⭐ PREPARE DATA FOR RAGAS
# ===========================================
ragas_data = {
    "question": [question],
    "answer": [answer],
    "contexts": [[doc.page_content for doc in retrieved_docs]],

    # ground truth answer required only for context_recall
    "ground_truths": ["Deep learning is a subset of ML using neural networks with many layers."]
}

dataset = Dataset.from_dict(ragas_data)


# ===========================================
# ⭐ RUN RAGAS METRICS
# ===========================================
result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        answer_relevancy,
        faithfulness
    ]
)

print("\n===================================")
print("📌 RAGAS METRICS")
print("===================================")
print(result.to_pandas())
