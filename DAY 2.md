# 📘 RAG Learning Notes --- Lesson 1

## Embeddings, Chunking, Batch Size & Reranking

> **Objective:** Learn how embeddings are generated, how documents are
> split into chunks, why normalization matters, what batch size does,
> and how reranking improves Retrieval-Augmented Generation (RAG).

------------------------------------------------------------------------

# Table of Contents

1.  What are Embeddings?
2.  Understanding `HuggingFaceEmbeddings`
3.  Embedding Normalization
4.  Batch Size
5.  Document Chunking
6.  Why Did We Get 50 Chunks?
7.  Reranking Models
8.  Embeddings vs Rerankers
9.  Using Embeddings and Rerankers Together
10. Complete RAG Pipeline
11. Best Practices
12. Interview Questions
13. Summary

------------------------------------------------------------------------

# 1. What are Embeddings?

An embedding is a numerical vector representation of text that captures
semantic meaning.

Example:

``` text
"I love AI"
↓
[0.214, -0.731, 0.502, ...]
```

Similar meanings produce similar vectors.

------------------------------------------------------------------------

# 2. HuggingFaceEmbeddings

``` python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={
        "batch_size":64,
        "normalize_embeddings":True
    }
)
```

  Parameter                 Meaning
  ------------------------- ----------------------------------------
  `HuggingFaceEmbeddings`   Loads the embedding model
  `model_name`              Specifies which embedding model to use
  `encode_kwargs`           Extra encoding parameters
  `batch_size`              Number of texts encoded simultaneously
  `normalize_embeddings`    Converts vectors to unit length

------------------------------------------------------------------------

# 3. Embedding Normalization

Without normalization:

``` text
[3,4]
Length = 5
```

With normalization:

``` text
[0.6,0.8]
Length = 1
```

Benefits:

-   Better cosine similarity
-   Consistent vector lengths
-   Faster similarity computation

------------------------------------------------------------------------

# 4. Batch Size

`batch_size=64` means the model encodes **64 text chunks at once**.

Example for 1000 chunks:

  Batch Size     Model Runs
  ------------ ------------
  1                    1000
  10                    100
  50                     20
  64                     16

**Batch size affects speed, not embedding quality.**

------------------------------------------------------------------------

# 5. Document Chunking

``` python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
```

-   `chunk_size` → Maximum characters per chunk
-   `chunk_overlap` → Shared characters between neighboring chunks

Example:

``` text
Chunk 1
|----------------1000----------------|

Chunk 2
              |----------------1000----------------|
              <-----150 overlap----->
```

------------------------------------------------------------------------

# 6. Why Did We Get 50 Chunks?

Approximate formula:

``` text
chunks ≈ ceil((DocumentLength-overlap)/(chunk_size-overlap))
```

For a document of about **42,500 characters**:

``` text
≈ ceil((42500-150)/850)
≈ 50 chunks
```

The exact count depends on:

-   Document length
-   Number of documents
-   Paragraph/newline boundaries
-   Chunk settings

------------------------------------------------------------------------

# 7. Reranking Models

A reranker reorders the documents returned by vector search.

``` text
Query
   │
   ▼
Top 20 Retrieved Documents
   │
   ▼
Reranker
   │
   ▼
Top 5 Most Relevant Documents
```

It reads the **query and each document together** and assigns a
relevance score.

------------------------------------------------------------------------

# 8. Embeddings vs Rerankers

  Feature                     Embedding   Reranker
  --------------------------- ----------- -----------------
  Output                      Vector      Relevance Score
  Speed                       Very Fast   Slower
  Searches Millions of Docs   ✅          ❌
  Accuracy                    Good        Excellent
  Purpose                     Retrieval   Reordering

------------------------------------------------------------------------

# 9. Using Both Together

Modern RAG systems combine both:

``` text
User Query
      │
      ▼
Embedding Model
      │
      ▼
Vector Database
      │
 Retrieve Top 20
      │
      ▼
Reranker
      │
 Keep Top 5
      │
      ▼
LLM
```

------------------------------------------------------------------------

# 10. Best Practices

-   Normalize embeddings when using cosine similarity.
-   Choose batch sizes based on available RAM/VRAM.
-   Use chunk overlap to preserve context.
-   Retrieve many documents, rerank, then send only the best to the LLM.

------------------------------------------------------------------------

# 11. Interview Questions

1.  What is an embedding?
2.  Why normalize embeddings?
3.  What does batch size control?
4.  Does batch size affect embedding quality?
5.  Why use chunk overlap?
6.  What is a reranker?
7.  Why are rerankers used after vector search?
8.  Why combine embeddings with reranking?

------------------------------------------------------------------------

# 12. Summary

-   Embeddings convert text into vectors.
-   Normalize vectors for cosine similarity.
-   Batch size improves throughput.
-   Chunk overlap preserves context.
-   Chunk count depends on document size.
-   Rerankers improve retrieval accuracy.
-   Production RAG = **Embedding Search + Reranker + LLM**.
