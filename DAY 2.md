# Today's Lesson: Embeddings, Chunking, Batch Size, and Reranking

## 1. Embeddings

Embeddings are numerical vector representations of text that capture
semantic meaning.

Example:

``` python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True}
)
```

-   `HuggingFaceEmbeddings`: Loads an embedding model.
-   `model_name="BAAI/bge-m3"`: Uses the BAAI BGE-M3 embedding model.
-   `encode_kwargs`: Extra options passed to the encoder.
-   `normalize_embeddings=True`: Produces unit-length vectors for cosine
    similarity.

## 2. Why Normalize Embeddings?

Cosine similarity compares vector directions, not magnitudes.

Benefits: - Better retrieval quality with cosine similarity. - Dot
product equals cosine similarity for normalized vectors. - Consistent
vector magnitudes.

## 3. Batch Size

Example:

``` python
encode_kwargs={
    "batch_size": 64,
    "normalize_embeddings": True
}
```

`batch_size=64` means the model encodes **64 texts at once**.

Advantages: - Faster processing. - Better GPU utilization.

Trade-off: - Larger batch size requires more memory. - It **does not
change embedding quality**.

## 4. Text Chunking

Example:

``` python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
```

-   `chunk_size=1000`: Maximum characters per chunk.
-   `chunk_overlap=150`: Consecutive chunks share 150 characters.

The number of chunks depends on: - Total document length. - Number of
documents. - Natural separators (paragraphs, newlines, etc.).

Example: - 42,500 characters with `chunk_size=1000` and
`chunk_overlap=150` produces about 50 chunks.

## 5. Reranking Models

A reranking model reorders the documents retrieved by the vector
database.

Pipeline:

``` text
User Query
    ↓
Embedding Model
    ↓
Vector Database (Top 20)
    ↓
Reranking Model
    ↓
Top 5 Documents
    ↓
LLM
```

### Embedding vs Reranker

  Embedding Model                  Reranking Model
  -------------------------------- ----------------------------------
  Converts text into vectors       Scores query-document pairs
  Very fast                        Slower
  Searches millions of documents   Reranks only retrieved documents
  Approximate retrieval            Highly accurate ordering

## 6. Can They Be Used Together?

Yes. This is the standard production RAG architecture.

1.  Embed documents.
2.  Store embeddings in a vector database.
3.  Embed the user query.
4.  Retrieve Top-K documents.
5.  Rerank those documents.
6.  Send the best documents to the LLM.

## Key Takeaways

-   Embeddings represent semantic meaning as vectors.
-   Normalize embeddings when using cosine similarity.
-   Batch size controls how many texts are encoded simultaneously.
-   Chunk count depends on document size, not just chunk settings.
-   Reranking improves retrieval accuracy after vector search.
-   Embeddings + reranking provide the best balance of speed and quality
    in RAG systems.
