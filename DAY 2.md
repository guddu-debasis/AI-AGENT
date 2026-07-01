````md
# 📘 RAG Learning Notes — Lesson 1
# Embeddings, Chunking, Batch Size & Reranking

> **Objective:** Learn how embeddings are generated, how documents are split into chunks, why normalization matters, what batch size does, and how reranking improves Retrieval-Augmented Generation (RAG).

---

# Table of Contents

1. What are Embeddings?
2. Understanding `HuggingFaceEmbeddings`
3. Embedding Normalization
4. Batch Size
5. Document Chunking
6. Why Did We Get 50 Chunks?
7. Reranking Models
8. Embeddings vs Rerankers
9. Using Embeddings and Rerankers Together
10. Complete RAG Pipeline
11. Best Practices
12. Interview Questions
13. Summary

---

# 1. What are Embeddings?

An **embedding** is a numerical representation (vector) of text that captures its semantic meaning.

Instead of storing words directly, an embedding model converts text into numbers.

Example:

Text:

```
"I love Machine Learning."
```

Embedding:

```text
[0.214, -0.731, 0.502, ..., 0.118]
```

Two sentences with similar meanings will have similar vectors.

Example:

```
"I love AI."

"I enjoy Artificial Intelligence."
```

These embeddings will be close together in vector space.

---

# 2. Understanding `HuggingFaceEmbeddings`

Example:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={
        "batch_size":64,
        "normalize_embeddings":True
    }
)
```

## Explanation

### `HuggingFaceEmbeddings`

Loads an embedding model from Hugging Face.

Its job is to convert text into vectors.

---

### `model_name="BAAI/bge-m3"`

Specifies which embedding model should be loaded.

- **BAAI** → Organization
- **bge-m3** → Embedding model

---

### `encode_kwargs`

Extra parameters passed to the model while generating embeddings.

Example:

```python
encode_kwargs={
    "batch_size":64,
    "normalize_embeddings":True
}
```

---

# 3. Embedding Normalization

```python
normalize_embeddings=True
```

This converts every embedding into a **unit vector**.

Without normalization:

```
Vector

[3,4]

Length = 5
```

After normalization:

```
[0.6,0.8]

Length = 1
```

Every vector has magnitude = 1.

---

## Why normalize?

Most vector databases use **Cosine Similarity**.

Cosine Similarity measures only the **direction** of vectors.

Normalization improves:

- Retrieval quality
- Faster similarity computation
- Consistent vector lengths

---

# 4. Batch Size

Example:

```python
batch_size=64
```

Batch size specifies **how many texts are encoded simultaneously.**

Suppose we have **1000 document chunks**.

### Batch Size = 1

```
Run 1 → Chunk 1

Run 2 → Chunk 2

Run 3 → Chunk 3

...

Run 1000
```

Total model runs:

```
1000
```

---

### Batch Size = 64

```
Run 1

Chunks 1–64

Run 2

Chunks 65–128

...

Run 16
```

Only about **16 runs** are required.

---

## Advantages

- Faster encoding
- Better GPU utilization
- Higher throughput

---

## Disadvantages

Larger batches require:

- More RAM
- More GPU memory (VRAM)

---

## Does Batch Size Affect Embedding Quality?

**No.**

Batch size only changes processing speed.

The embedding values remain the same.

---

# 5. Document Chunking

Example:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
```

---

## `chunk_size`

Maximum characters in each chunk.

Example:

```
1000 characters
```

---

## `chunk_overlap`

Characters shared between consecutive chunks.

Example:

```
Chunk 1

|----------------1000----------------|

Chunk 2

              |----------------1000----------------|

              <------150 overlap------>
```

This overlap preserves context across chunk boundaries.

---

# 6. Why Did We Get 50 Chunks?

Suppose:

```
Document Length

42,500 characters
```

Settings:

```
chunk_size = 1000

chunk_overlap = 150
```

Effective movement:

```
1000 − 150

=

850 characters
```

Approximate formula:

```
Chunks

≈ ceil(

(Document Length − Overlap)

/

(chunk_size − overlap)

)
```

Example:

```
≈ ceil(

(42500−150)

/850

)

≈ 50 chunks
```

---

### Important

The number of chunks depends on:

- Document length
- Number of documents
- Paragraph breaks
- Newlines
- Chunk size
- Chunk overlap

It **cannot** be determined from `chunk_size` alone.

---

# 7. Reranking Models

A reranking model improves retrieval quality by **reordering retrieved documents**.

Imagine the vector database returns:

| Rank | Document |
|------|----------|
| 1 | Account Creation |
| 2 | Password Reset |
| 3 | Login Issues |

User asks:

```
How do I reset my password?
```

The best document is actually **Password Reset**.

A reranker fixes this.

After reranking:

| Rank | Document |
|------|----------|
| 1 | Password Reset ✅ |
| 2 | Login Issues |
| 3 | Account Creation |

---

## How does a reranker work?

Instead of comparing vectors,

it compares:

```
Query

+

Document

↓

Relevance Score
```

Example:

```
Query

How do I reset my password?

Document

Click "Forgot Password"...

↓

Score

0.98
```

---

# 8. Embeddings vs Rerankers

| Feature | Embedding Model | Reranking Model |
|----------|-----------------|-----------------|
| Input | One text | Query + Document |
| Output | Vector | Relevance Score |
| Speed | Very Fast | Slower |
| Searches Millions of Docs | ✅ | ❌ |
| Accuracy | Good | Excellent |
| Used For | Initial Retrieval | Final Ranking |

---

# 9. Can We Use Both Together?

**Yes!**

Modern RAG systems always combine them.

Workflow:

```
User Query

↓

Embedding Model

↓

Vector Database

↓

Top 20 Documents

↓

Reranker

↓

Top 5 Documents

↓

LLM

↓

Final Answer
```

---

# 10. Complete RAG Pipeline

```
Documents

↓

Chunking

↓

Embedding Model

↓

Vector Database

────────────────────────────

User Query

↓

Embedding

↓

Vector Search

↓

Top-K Documents

↓

Reranker

↓

Best Documents

↓

LLM

↓

Answer
```

---

# 11. Best Practices

### Embeddings

✅ Normalize embeddings for cosine similarity.

---

### Chunking

- Keep chunk size between **500–1000 characters** (or an equivalent token budget, depending on your model).
- Use **10–20% overlap** to preserve context.

---

### Batch Size

- CPU → 8–16
- GPU → 32–128 (depending on available VRAM)

---

### Reranker

Retrieve many documents first.

Example:

```
Retrieve Top 20

↓

Rerank

↓

Keep Top 5
```

---

# 12. Interview Questions

### Beginner

1. What is an embedding?

2. Why are embeddings needed?

3. What is cosine similarity?

4. Why normalize embeddings?

5. What does `batch_size` do?

---

### Intermediate

6. Does batch size affect embedding quality?

7. What is chunk overlap?

8. Why do we split documents?

9. How is chunk count calculated?

10. What is a reranking model?

---

### Advanced

11. Why can't rerankers search millions of documents?

12. Why are rerankers more accurate?

13. Explain a complete RAG retrieval pipeline.

14. Difference between an embedding model and a cross-encoder reranker?

15. Why do production RAG systems use both?

---

# 13. Summary

✅ Embeddings convert text into vectors.

✅ Normalize embeddings when using cosine similarity.

✅ Batch size controls how many texts are encoded together.

✅ Chunk overlap preserves context.

✅ The number of chunks depends on document size and separators.

✅ Rerankers improve retrieval accuracy by reordering retrieved documents.

✅ Modern RAG systems combine:

```
Embedding Model
        +
Vector Database
        +
Reranker
        +
LLM
```

to achieve both **high speed** and **high retrieval accuracy**.
````
