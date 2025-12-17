Great! You already have a **working RAG pipeline**, which means you’re now ready to:

### ✔ Evaluate your **retrieval quality** using retrieval metrics (Recall@K, Precision@K, etc.)

### ✔ Evaluate your **full RAG pipeline** using **RAGAS metrics** (context precision, recall, faithfulness, answer relevancy)

I will show you **exactly how to connect RAGAS to your existing code**, step by step.

---

# ⭐ **PART 1 — Add Retrieval Metrics (Recall, Precision, MRR, nDCG, Hit Rate)**

First, copy/paste the retrieval metric functions into your notebook.

```python
import math

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
    ideal = sorted(
        list(relevant_docs),
        key=lambda d: rel_scores[d],
        reverse=True
    )
    return dcg(retrieved, rel_scores) / dcg(ideal, rel_scores)
```

---

# ⭐ **PART 2 — Add metadata IDs to your chunks**

FAISS needs doc IDs for retrieval metrics.

Modify your chunk creation:

```python
for i, chunk in enumerate(chunks):
    chunk.metadata["id"] = str(i)
```

---

# ⭐ **PART 3 — Retrieve Docs & Compute Retrieval Metrics**

After retrieval:

```python
question = "explain alien?"
retrieved_docs = retriever.invoke(question)
```

Assume **relevant_docs** are known (for testing):

```python
relevant_docs = {"3", "5"}   # example - depends on your dataset
```

Define relevance scores (for nDCG):

```python
relevance_scores = {"3": 3, "5": 2}
```

Now compute metrics:

```python
print("Recall@5 :", recall_at_k(retrieved_docs, relevant_docs, 5))
print("Precision@5 :", precision_at_k(retrieved_docs, relevant_docs, 5))
print("MRR :", mrr(retrieved_docs, relevant_docs))
print("Hit Rate:", hit_rate_at_k(retrieved_docs, relevant_docs, 5))
print("nDCG:", ndcg(retrieved_docs, relevant_docs, relevance_scores))
```

---

# ⭐ **PART 4 — Implement RAGAS on Your Pipeline**

## ✔ Install RAGAS

```python
pip install ragas
pip install datasets
```

---

# ⭐ **STEP 1 — Generate Answers using Your RAG Pipeline**

We need:

* question
* answer
* contexts
* ground_truth (optional)

Your generator:

```python
final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = llm.invoke(final_prompt).content
```

---

# ⭐ **STEP 2 — Prepare Dataset for RAGAS**

RAGAS expects:

* question (string)
* answer (string)
* contexts (list of strings)
* ground_truths (list of strings) → optional

```python
from datasets import Dataset

data = {
    "question": [question],
    "answer": [answer],
    "contexts": [[doc.page_content for doc in retrieved_docs]],
    "ground_truths": [["I don't know"]]   # or real ground truth if available
}

dataset = Dataset.from_dict(data)
```

---

# ⭐ **STEP 3 — Run RAGAS Evaluation**

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness
)

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        answer_relevancy,
        faithfulness
    ]
)

result_df = result.to_pandas()
result_df
```

---

# ⭐ **UNDERSTANDING RAGAS OUTPUT**

RAGAS gives you:

| Metric                | Meaning                              |
| --------------------- | ------------------------------------ |
| **context_precision** | Is retrieved context relevant?       |
| **context_recall**    | Did retrieval fetch all needed info? |
| **answer_relevancy**  | Does answer match question?          |
| **faithfulness**      | Is answer grounded in context?       |

You get values between **0 and 1**.

Example:

| context_precision | context_recall | answer_relevancy | faithfulness |
| ----------------- | -------------- | ---------------- | ------------ |
| 0.74              | 1.0            | 0.88             | 0.65         |

---

# ⭐ **FULL END-TO-END PIPELINE SUMMARY**

You now have:

### ✔ A complete basic RAG pipeline

### ✔ Retrieval evaluation metrics (Recall, Precision, MRR, nDCG, HitRate)

### ✔ RAGAS evaluation (context quality + answer quality)

This is **production-quality evaluation**.

---

# ⭐ If you want, I can also provide:

📌 A unified RAG evaluation script that prints all metrics at once
📌 A visual diagram of your RAG pipeline + RAGAS placement
📌 How to tune your retriever using RAGAS metrics
📌 A ready-to-run Google Colab notebook

Just tell me!
