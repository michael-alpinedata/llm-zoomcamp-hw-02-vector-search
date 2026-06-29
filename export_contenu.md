# Architecture du projet : /home/micha/learning/llm-zoomcamp-hw-02-vector-search
```text
/home/micha/learning/llm-zoomcamp-hw-02-vector-search
├── README.md
├── code
│   ├── vector_search.ipynb
│   ├── vector_search_persistent.ipynb
│   └── vector_search_pgvector.ipynb
├── download.py
├── embedder.py
├── export_contenu.md
├── homework_2.ipynb
├── homework_2.md
├── lessons
│   ├── 01-intro.md
│   ├── 02-embeddings.md
│   ├── 03-embeddings-dataset.md
│   ├── 04-vector-search.md
│   ├── 05-minsearch-vector.md
│   ├── 06-rag-vector.md
│   ├── 07-sqlitesearch-vector.md
│   ├── 08-pgvector.md
│   ├── 09-onnx-embedder.md
│   └── 10-next-steps.md
├── models
│   └── Xenova
│       └── all-MiniLM-L6-v2
│           ├── model.onnx
│           └── tokenizer.json
├── pyproject.toml
└── uv.lock

6 directories, 23 files
```

---

## Fichier : lessons/07-sqlitesearch-vector.md
```text
# Vector Search with sqlitesearch

Video: [Watch this lesson](https://www.youtube.com/watch?v=csxKescwJYM&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

In the previous section we used minsearch for vector search.

It works, but it has three problems:

- It rebuilds the index on every startup
- It keeps everything in memory
- It searches by brute force


With text search we never felt these. Indexing was fast because we
didn't embed anything. With vector search, indexing runs a neural
network over every document, so it takes a minute on our dataset.
Keeping everything in memory is fine here, but a larger dataset would
need too much space.

The third problem is brute-force search. For every query we compare the
query vector against every single document. With 1,000 documents this is
fine, probably even faster than anything smarter. But as the dataset
grows past 10,000 or so, it gets slow, and we'll want an approximate
method instead.

What we've done so far is exact nearest neighbor (NN) search. We score
the query against every document and pick the top ones. It always finds
the true top matches, but it pays for that by touching everything.

Approximate nearest neighbor (ANN) search takes a shortcut. Instead of
comparing against everything, it first narrows down to a region of
likely matches. Then it scores only within that region. It may miss the
absolute best match, but the results are still good and it's much
faster.

```text
NN (exact):    compare query against ALL documents -> top 5
ANN (approx):  narrow down to a region -> compare within region -> top 5
```

## sqlitesearch

sqlitesearch is the persistent sibling of minsearch, and it solves both
problems at once.

We already used it in module 1 for persistent text search. It also does
vector search through its `VectorSearchIndex` class. It stores vectors
in SQLite, a real on-disk database, and uses ANN strategies for
retrieval. Because the data lives on disk, one process can write the
vectors and another can read them back.

If you didn't install it in the previous module, add it to your project:

```bash
uv add sqlitesearch
```

## Creating the index

Initialize it:

```python
from sqlitesearch import VectorSearchIndex

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```

sqlitesearch supports three ANN modes:

- `lsh` (default): up to 100K vectors, random hyperplane projections
- `ivf`: 10K-500K vectors, K-means clustering
- `hnsw`: 10K-1M+ vectors, proximity graph (highest recall)

For our small dataset, `lsh` is fine. All modes use two-phase search:
approximate candidate retrieval, then exact cosine similarity
reranking.

## Indexing the data

Fit the index with our vectors and documents:

```python
vs_index.fit(vectors, documents)
```

The index is saved to `faq_vectors2.db`. Unlike minsearch, this file
persists on disk. You can search immediately after indexing, or reopen
the index later without re-indexing.

## Searching

Search works the same way as with minsearch. We always encode the query
into a vector first. This is one thing that makes vector search heavier
than text search. With text search we'd throw the raw query straight at
the engine.

Encode, then search:

```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vs_index.search(query_vector, num_results=5)
```

Look at the results:

```python
results
```

## Filtering by course

Filtering works the same way:

```python
results = vs_index.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

## Closing the connection

When you're done with the index:

```python
vs_index.close()
```


## Reopening the index

In a new Python session, you can reopen the index without re-computing
embeddings:

```python
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

model = SentenceTransformer("all-MiniLM-L6-v2")

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```

Now we can search:

```python
query_vector = model.encode("How do I run Kafka?")
results = vs_index.search(query_vector, num_results=5)
```

We still load the embedding model to encode the query, but we don't
re-embed all the documents. No `fit` call needed, because the index is
already built and waiting on disk.

This is the same two-process split we used for text search in module 1.
One process ingests and builds the index, another queries it.

It matters more here than with text search. Embedding the whole dataset
takes about a minute. We don't want a user waiting that long when the
app starts up. We pay that cost once during ingestion, and the query
side starts up instantly.

## Using sqlitesearch vector search in RAG

Let's use our persistent vector index in RAG.

In a new notebook, set up the model and open the index (same as
the "Reopening the index" section above):

```python
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

model = SentenceTransformer("all-MiniLM-L6-v2")

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```

We'll use the `RAGVector` class we defined in the
[previous lesson](06-rag-vector.md). It overrides the `search` method
to embed the query and use vector search.

Set up the OpenAI client and create the assistant:

```python
from rag_helper import RAGBase
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )

vector_assistant = RAGVector(
    embedder=model,
    index=vs_index,
    llm_client=openai_client,
)
```

Try it:

```python
vector_assistant.rag("the program has already begun, can I still sign up?")
```

When you're done, close the connection:

```python
vs_index.close()
```

## Comparing minsearch and sqlitesearch for vector search

Here is how the two compare:

- minsearch `VectorSearch`: in-memory (numpy), exact cosine similarity,
  must re-compute embeddings on startup, good for experiments and
  notebooks
- sqlitesearch `VectorSearchIndex`: persistent (SQLite `.db` file), ANN
  (LSH/IVF/HNSW) with exact rerank, can open an existing index, good
  for projects and persistence

This is probably the last you'll hear of sqlitesearch. I built it for
teaching, to show the ingestion-then-deployment split.

It does have a real use though. Its only dependencies are SQLite and
numpy. So it runs on any host that offers a free SQLite database, where
a dedicated vector database would cost extra. For most work you'll reach
for something else, which is what we do next.

[← RAG with Vector Search](06-rag-vector.md) | [Vector Search with PGVector →](08-pgvector.md)

```

## Fichier : lessons/03-embeddings-dataset.md
```text
# Embedding Our Dataset

Video: [Watch this lesson](https://www.youtube.com/watch?v=NC89mz1iG4E&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

In the previous lesson we saw how embeddings work on a couple of
examples. Now we apply them to the whole FAQ dataset.

## Loading the data

In [module 1](../../01-agentic-rag/) we created
[`ingest.py`](../../01-agentic-rag/code/ingest.py) for loading the
FAQ data.

Download it into your project:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/ingest.py
```

We use it here:

```python
from ingest import load_faq_data

documents = load_faq_data()
```

## Generating embeddings

Each document is a Python dictionary with a question and an answer. We
embed both together. That way a query can match against the question
text and the answer text in our index.

Build one text per document:

```python
texts = []

for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)
```

Now we generate the embeddings. We have about 1200 texts here. We won't
hand the model all of them at once. That takes a long time, and we can't
see what's happening inside. Instead we split them into batches.

First we import `tqdm` so we can watch the progress:

```python
from tqdm.auto import tqdm
```

Next we chunk the dataset into batches of 50 and encode each batch:

```python
batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

len(vectors)
```

We end up with 1208 vectors. On a GPU this is fast. Most of us run on
Codespaces without a GPU, so it takes a bit, but it's a one-off.

We turn them into a 2-dimensional array (matrix) where

- rows are documents (vectors)
- columns are dimensions of the vectors

```python
import numpy as np
X = np.array(vectors)
```

Calling `X.shape` returns (1208, 384) - number of documents vs number of dimensions.

[← Embeddings](02-embeddings.md) | [Vector Search →](04-vector-search.md)

```

## Fichier : lessons/05-minsearch-vector.md
```text
# Vector Search with minsearch

Video: [Watch this lesson](https://www.youtube.com/watch?v=E7KdO3xmESg&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

In the previous section we did vector search by hand with numpy. We
embedded the query, computed dot products, and found the best matches.
Writing the argsort and matrix code every time gets old, and it can't
filter by course. So instead we'll use a library that wraps all of it.

We'll use [minsearch](https://github.com/alexeygrigorev/minsearch), the
small in-memory search library we already used in module 1 for text
search. It has a `VectorSearch` class for vector search.

Both classes share the same API:

- `fit` to index data
- `search` to query
- `filter_dict` in `search` to filter by keyword

It's the simplest way to get started with vector search.

## Creating the index

We already have our documents and vectors from the previous section.

Index them:

```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)
```

We pass the numpy array `X` with all embeddings and the list of
documents as payload. The `keyword_fields` parameter works the same as
in the text `Index`, so we can filter by course later.

## Searching

Let's search for a question:

```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vindex.search(query_vector, num_results=5)
```

Under the hood it does the same thing we just did by hand. It computes
the dot product between each vector (after filtering) and our query
vector.

Look at the top result:

```python
results[0]
```

It should return the document about joining the course late:


```python
{"id": "74eb249bbf",
 "course": "llm-zoomcamp",
 "section": "General Course-Related Questions",
 "question": "I just discovered the course. Can I still join?",
 "answer": "Yes, but if you want to receive a certificate, you need to submit your project while we’re still accepting submissions."}
```

## Filtering by course

Like the text index, we can filter by keyword fields. This matters for
user experience. A student in LLM Zoom Camp doesn't care about answers
from the data engineering course. So we narrow to their course first,
then score only within it.

Pass a `filter_dict`:

```python
results = vindex.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

Now that we can run vector search, let's use it in RAG.

[← Vector Search](04-vector-search.md) | [RAG with Vector Search →](06-rag-vector.md)

```

## Fichier : lessons/10-next-steps.md
```text
# Next Steps

Video: [Watch this lesson](https://www.youtube.com/watch?v=vhNfnNUz3A0&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

In this module, we:

- Learned what embeddings are and how they turn text into vectors
- Generated embeddings for our FAQ dataset using sentence-transformers
- Built vector search with numpy, minsearch, sqlitesearch, and PGVector
- Integrated vector search into our RAG pipeline with the `RAGVector`
  class

The code is available in the
[code directory](../code/).


## Using vector search

Text search is simple and fast, and for many applications it's all you
need. Start there.

Vector search adds real overhead:

- You need an embedding model
- You need to compute and store embeddings
- At query time you encode the query before you can search

Even with the smallest model that overhead is considerable, and that's
before counting the extra dependencies. Don't take it on without a
reason.

Most RAG tutorials assume you need vector search from the start. Quite a
few of them come from companies that sell vector databases. So of course
they push it. You don't need it on day one.

A reasonable progression:

1. v1: Start with text search. Get your RAG pipeline working end to end.
   For our FAQ dataset, text search already handles most questions well.
   We saw that in module 1.
2. v2: Add vector search when you can measure that text search misses
   relevant results. This happens when users ask questions in different
   words than what is in your documents.
3. v3: Combine both with hybrid search (text + vector), which typically
   outperforms either one alone.

The right time to move from one version to the next is when evaluation
shows it's justified. A later module covers how to evaluate search
quality and compare methods. With numbers in hand, you can tell a
marginal gain from a real one. A marginal gain isn't worth the overhead.
A real one is.


## Hybrid search and reranking

Once you have both text search and vector search, you can combine them.
Hybrid search runs both methods and merges the results using techniques
like Reciprocal Rank Fusion (RRF).

Reranking takes it further. After retrieving candidate documents, a
separate model re-scores them for relevance.

Both techniques can improve retrieval quality. We cover them in the
[Hybrid Search lesson in the Best Practices module](../../06-best-practices/lessons/02-hybrid-search.md).


## Moving forward

Try these next steps:

- Compare text search and vector search on your own data
- Experiment with different embedding models
- Try PGVector with a larger dataset
- Try a different vector database - Elasticsearch, Qdrant,
  Weaviate, Chroma, or any other. The concepts are the same:
  embed documents, store vectors, search by similarity
- Evaluate your search results (covered in a later module)

[← Using ONNX Runtime](09-onnx-embedder.md) | [Back to module →](../)

```

## Fichier : lessons/04-vector-search.md
```text
# Vector Search

Video: [Watch this lesson](https://www.youtube.com/watch?v=h-_tdBc24qc&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

In the previous lesson we embedded our FAQ dataset into a matrix `X`
with 1208 document vectors. Here we see how vector search works under
the hood.

## Scoring documents

We have a matrix `X` with all document embeddings. We take a query,
compare it against every document, and keep the most similar ones.

When a query comes in, we embed it:

```python
query = "Can I still join the course after the start date?"
v_query = model.encode(query)
```

Next, we compute the dot product against all documents:

```python
scores = X.dot(v_query)
```

This is matrix-vector multiplication. Each element `i` of `scores` is
the cosine similarity between document `i` (row `i` of `X`) and
`v_query`.

We could compute the same thing with a for loop:

```python
scores = [v_query.dot(X[i]) for i in range(len(X))]
```

But `X.dot(v_query)` is much faster. Numpy runs optimized C code instead
of looping in Python, so the matrix version is hard to beat. The outcome
is the same either way: one score per document.

## Best match

The highest score is the most similar document:

```python
idx = np.argmax(scores)
idx, scores[idx]
```

This returns document 553 with score 0.76.

The index and score may differ for you. Our FAQ is a living document, so
we add and remove entries over time.

Let's see which document it is:

```python
documents[idx]
```

We see:

```python
{"id": "3f1424af17",
 "course": "data-engineering-zoomcamp",
 "section": "General Course-Related Questions",
 "question": "Course: Can I still join the course after the start date?",
 "answer": "Yes, even if you don't register, you're still eligible..."}
```

## Top 5 results

Usually we want more than the single best match, so let's pull the top
5.

`np.argsort` sorts from lowest to highest, so the last 5 are the top
ones:

```python
top5 = np.argsort(scores)[-5:]
```

They come out smallest-first, so we reverse them to get the highest
first:

```python
top5 = top5[::-1]
top5
```

Now we can read off the top 5 scores:

```python
scores[top5]
```

There's a shorter trick I usually reach for. We negate the scores
first, so the largest becomes the smallest. Then `argsort` puts the best
matches at the front.

Here it is in one line:

```python
top5 = np.argsort(-scores)[:5]
```

It looks cryptic the first time you see it. But it's a common way to
turn a min-sort into a max-sort.

Let's read off the actual documents:

```python
for idx in top5:
    print(scores[idx])
    print(documents[idx])
    print()
```

This is vector search in its simplest form. We embed the query, compute
dot products against all documents, and return the highest-scoring ones.

We return 5 and not the single best for a reason. The answer to a
question can be spread across several documents. One holds part of it,
another fills in the rest. Sometimes the top result isn't the right one
but the second is. We send all 5 to the LLM and let it combine them.

The number 5 is a starting point, picked on gut feeling. Later, when we
evaluate search quality, we can test whether 3 or 10 works better for
our data.

Doing this by hand with numpy is fine for a small dataset. A larger one
needs a library that also handles filtering and ranking. That's what we
turn to next.

[← Embedding Our Dataset](03-embeddings-dataset.md) | [Vector Search with minsearch →](05-minsearch-vector.md)

```

## Fichier : lessons/06-rag-vector.md
```text
# RAG with Vector Search

Video: [Watch this lesson](https://www.youtube.com/watch?v=-GBW3g3PVTM&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

In module 1, we built a RAG pipeline with three steps:

```python
def rag(question):
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)
```

The search step used keyword search. Now we swap in vector search.
Because RAG is modular, search is the only step we touch. Build prompt
and the LLM call stay exactly as they were.

## Using RAGBase

In [module 1](../../01-agentic-rag/) we put all the RAG logic into a
[`RAGBase`](../../01-agentic-rag/code/rag_helper.py) helper class. It
has `search`, `build_prompt`, and `llm` methods, so we only need to
override `search`.

Download `rag_helper.py` (and `ingest.py` if you didn't get it earlier)
into your project:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/rag_helper.py
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/ingest.py
```

First, create the OpenAI client:

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()
```

Next, download and index the data:

```python
from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)
```

Then use the `RAGBase` class:

```python
from rag_helper import RAGBase

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
)
```

Ask it a question:

```python
query = "I just found out about the program, can I still sign up?"
assistant.rag(query)
```

This still uses keyword search. Text search isn't bad here, so the
answer may already look right. Next we replace search with vector
search.

We already have:

- All the indexed documents `documents`
- The embeddings matrix `X` with all these documents
- The vector search engine `vindex`

We can't pass `vindex` to RAG as-is. Text search takes the query string
directly, but vector search needs the query as a vector first. So we
subclass `RAGBase` and override `search` to encode the query before
searching.

The subclass overrides `search`:

```python

class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )
```

The `__init__` method adds one extra argument, `embedder`, for the
sentence transformer. Inside `search` we use it to turn the query into a
vector. Then we query `vindex` with that vector instead of the raw text.
Everything else is inherited from `RAGBase`.

## Using it

Let's init it:

```python
vector_assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=openai_client,
)
```

Try it with different queries:

```python
vector_assistant.rag("the program has already begun, can I still sign up?")
```

The answers should be close to what we got with keyword search, but
vector search handles rephrased questions better. The swap was trivial
because RAG has three clear steps. The same trick lets us change the LLM
provider later by overriding just the `llm` step.

[← Vector Search with minsearch](05-minsearch-vector.md) | [Vector Search with sqlitesearch →](07-sqlitesearch-vector.md)

```

## Fichier : lessons/08-pgvector.md
```text
# Vector Search with PGVector

Video: [Watch this lesson](https://www.youtube.com/watch?v=0P54MFyz-mc&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

Many real databases can do vector search. Elasticsearch has it, and
there are dedicated stores like Qdrant and Chroma. We'll go with
Postgres. Most of us already run it at work, and the data engineering
course uses it too. The concept is the same as with sqlitesearch, only
the database under the hood changes.

[pgvector](https://github.com/pgvector/pgvector) is the PostgreSQL
extension that makes this work. Install it and Postgres can do vector
similarity search. On top of that you get the usual production features,
like concurrent access, transactions, and large datasets.

We'll run Postgres with pgvector in Docker.

## Starting Postgres with pgvector

Pull the image and start a container:

```bash
docker run -it \
    --name pgvector \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=faq \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17
```

This image has the pgvector extension pre-installed. The `-v` flag
creates a named volume so data persists across container restarts.

## Installing the Python client

Install the driver:

```bash
uv add psycopg[binary]
```

Note: if using Zshell use `uv add 'psycopg[binary]'`

We'll use `psycopg` (v3) to connect and run queries. Note: this is
different from `psycopg2` - psycopg v3 supports `conn.execute()`
directly without creating a cursor.

## Preparing the data

We need the FAQ documents and their embeddings.

Here's what we did in previous units as one script:

```python
from tqdm.auto import tqdm

from ingest import load_faq_data
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_faq_data()
texts = [doc["question"] + " " + doc["answer"] for doc in documents]

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)
```


Now we connect to Postgres:

```python
import psycopg

conn = psycopg.connect(
    "postgresql://user:pswd@localhost:5432/faq"
)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
```

The second line activates pgvector. The Docker image we started isn't
plain Postgres, it ships the extension inside, and this turns it on. It
adds the `vector` column type and the similarity search operators.

## Creating a table

Create a table for storing documents with their embeddings:

```python
conn.execute("""
    DROP TABLE IF EXISTS documents
""")

conn.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        course TEXT,
        section TEXT,
        question TEXT,
        answer TEXT,
        embedding vector(384)
    )
""")
```

The `vector(384)` column stores our 384-dimensional embeddings from
`all-MiniLM-L6-v2`.

## Inserting documents with embeddings

Let's insert the documents and their vectors into PGVector:

```python
def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"

for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (doc["course"], doc["section"], doc["question"], doc["answer"],
         vec_to_str(vec))
    )

conn.commit()
```

We loop over the documents and insert each one with its embedding. We
hand Postgres the vector as text, so the `::vector` cast tells it to
parse that string back into a vector. We call `conn.commit()` to persist
the changes.

## Searching with cosine similarity

Search with a query:

```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)
query_str = vec_to_str(query_vector)
```

Search for the most similar documents:

```python
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, query_str)
).fetchall()

for row in results:
    print(f"[{row[0]}] {row[1]} (similarity: {row[3]:.4f})")
```

The `<=>` operator computes cosine distance (1 - cosine similarity).
We order by ascending distance, so the closest vectors come first.

## Filtering by course

Because this is plain SQL, filtering by course is one extra `WHERE`
clause:

```python
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE course = %s
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, "llm-zoomcamp", query_str)
).fetchall()
```

## Creating an index for faster search

So far this runs brute-force search, comparing our query against every
row. For our small dataset that's fine.

For a larger one, create an HNSW index to switch to approximate search:

```python
conn.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
""")
```

This builds an HNSW (Hierarchical Navigable Small World) index, the
same state-of-the-art algorithm dedicated vector databases use. It makes
search faster, at the cost of a small accuracy trade-off.

## Wrapping it in a function

Let's wrap the search logic in a reusable function:

```python
def pgvector_search(query, course="llm-zoomcamp", num_results=5):
    query_vector = model.encode(query)
    query_str = vec_to_str(query_vector)
    rows = conn.execute(
        """
        SELECT course, section, question, answer
        FROM documents
        WHERE course = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (course, query_str, num_results)
    ).fetchall()

    return [
        {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
        for r in rows
    ]
```

Try it:

```python
results = pgvector_search("How do I join the course?")
```

## Using it in RAG

We take the same `search` function from above and move it into a class.
We pass the Postgres connection instead of an index. We set `index=None`
because `RAGBase` expects an index and would complain otherwise.

The class overrides `search` to query PGVector:

```python
from rag_helper import RAGBase

class RAGPgVector(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]
```

Initialize OpenAI client:

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()
```

Create the vector assistant:

```python
vector_assistant = RAGPgVector(
    embedder=model,
    conn=conn,
    llm_client=openai_client,
)
```

Try it:

```python
vector_assistant.rag("the program has already begun, can I still sign up?")
```

## Using PGVector

Here's how PGVector compares with the two tools we used earlier:

- minsearch: no setup, in-memory, best for notebooks and experiments
- sqlitesearch: no setup, SQLite file persistence, best for pet projects
- PGVector: requires Docker, Postgres database with concurrent access,
  handles millions of records, best for production systems

Reach for PGVector when you need production features:

- concurrent reads and writes
- transactions
- integration with an existing Postgres-based application

[← Vector Search with sqlitesearch](07-sqlitesearch-vector.md) | [Using ONNX Runtime →](09-onnx-embedder.md)

```

## Fichier : lessons/01-intro.md
```text
# Vector Search

Video: [Watch this lesson](https://www.youtube.com/watch?v=qyZgxTmC2cY&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

In module 1 we used keyword search with minsearch and sqlitesearch.
It matches exact words. If you search for "Docker", the document has
to contain "Docker" to come back.

But look at these two questions:

- "Can I still join the course after the start date?"
- "Is it possible to enroll late?"

They mean the same thing, yet they share almost no words. A keyword
engine struggles to match them. We need something that works on
meaning, not on the exact words.

That something is vector search. Instead of matching words, it matches
ideas.

## The vector search process

We run vector search in two stages.

1. Offline (indexing): we convert all documents into vectors (arrays
   of numbers) and store them in an index.
2. Online (querying): we convert the user's query into a vector with
   the same model, then find the closest document vectors by similarity.

An embedding model produces these vectors. It's a neural network
trained to capture meaning, so texts that mean similar things land on
similar vectors. We measure how close two vectors are with a distance
metric. The most common one is cosine similarity.

Cosine similarity measures the angle between two vectors:

- Vectors pointing in the same direction: similarity close to 1
  (similar)
- Vectors at right angles: similarity close to 0 (unrelated)
- Vectors pointing in opposite directions: similarity close to -1
  (opposite meaning)

The larger the cosine similarity, the more similar the two texts are
in meaning.

## Keyword search vs vector search

Here's how the two approaches differ:

- Keyword search matches exact words. Vector search matches meaning.
- Keyword search suits specific terms, IDs, and names. Vector search
  suits paraphrased questions and natural language.
- Keyword search example: "pandas dataframe". Vector search example:
  "How do I work with tabular data?"
- Keyword search uses an inverted index (BM25, TF-IDF). Vector search
  uses a vector index based on cosine similarity.
- Keyword search misses synonyms and paraphrases. Vector search misses
  exact term matches.

Vector search is usually better, but it adds a lot of operational
complexity, and you'll feel that throughout this module. So my advice
is to never start with vector search. Start with text search, and reach
for vectors once you can show they're worth the extra cost.

In practice the two work best together. Hybrid search combines them,
and we cover it in the
[Best Practices module](../../06-best-practices/lessons/02-hybrid-search.md).

## Building vector search

We'll take the same FAQ dataset from module 1 and build vector search
with three tools:

1. minsearch - in-memory vector search (simplest, good for
   experiments)
2. sqlitesearch - persistent vector search backed by SQLite
   (production-friendly, same API as minsearch)
3. PGVector - vector search in PostgreSQL (scalable, runs in
   Docker)

Then we'll plug vector search into our RAG pipeline.

The code from this module is available in the
[code directory](../code/).

## Prerequisites

In module 1 we set up a project with several libraries. Here we also
install sentence-transformers. It pulls in PyTorch and is heavy, so I
recommend a fresh project (a separate Codespace) for this module alone.

If you're starting from scratch, create a new project and install the
module 1 libraries:

```bash
mkdir llm-zoomcamp-code
cd llm-zoomcamp-code
uv init
uv add requests minsearch openai jupyter python-dotenv
```

You also need a `.env` file with your API key. See the
[module 1 environment setup](../../01-agentic-rag/lessons/02-environment.md)
for details.

[← Back to module](../) | [Embeddings →](02-embeddings.md)

```

## Fichier : lessons/02-embeddings.md
```text
# Embeddings

Video: [Watch this lesson](https://www.youtube.com/watch?v=kJOlW1HeMp4&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

Before we can do vector search, we need to turn our text into vectors.
We call this process embedding: we embed text into a vector space. The
vectors we get back are also called "embeddings."

## Word embeddings and sentence embeddings

This idea comes from
[word2vec](https://en.wikipedia.org/wiki/Word2vec). The model learns to
place words as points in a multi-dimensional space. Words with similar
meanings land close to each other.

Imagine a 2D space where "enroll" and "join" are near each other and
"Docker" is far away:

```text
        · enroll
       · join
                   · Docker
```

The same idea works for entire sentences:

```text
Q1: "I just discovered the course. Can I still join it?"
Q2: "I just found out about the program. Can I still enroll?"

These two are close - they mean the same thing.

Q3: "How do I run Docker on Windows?"

This one is far away from Q1 and Q2.
```

Now imagine all 1200 documents in our FAQ dataset. Each one becomes a
point in this space. When a user asks a question, we embed it into the
same space and find the closest documents. Those nearest neighbors are
our search results.

The model encodes the whole sentence, not the words in isolation. So it
can tell apart the same word in different contexts.

Take the word "judge." In "the judge ruled out the possibility of crime"
(legal) it gets one vector. In "LLM-as-a-judge approach to evaluate
LLMs" (ML evaluation) it gets a different one. The surrounding context
changes the embedding.

So an embedding model takes text in and returns a fixed-length array of
numbers. We train it so that texts with similar meanings get similar
vectors.

We'll use [sentence-transformers](https://www.sbert.net/), a popular
open-source library for embeddings. It runs locally on your machine, so
there are no API costs.

## Installing sentence-transformers

Install the library:

```bash
uv add sentence-transformers
```

This also pulls in PyTorch under the hood, so it downloads a lot. You'll
see CUDA and other Nvidia packages go by. That's fine for experiments,
and we'll trim it down for production in a
[later lesson](09-onnx-embedder.md).

## Choosing a model

Sentence-transformers supports many models. The right one depends on
your task, your language, and the resources you have. Larger models are
usually slower, so for our FAQ dataset of short English texts a small
model is enough. Try a few on your own data and keep the one that works
best.

We'll use `all-MiniLM-L6-v2`:

- 384-dimensional vectors (compact)
- Fast on CPU
- Good quality for general English text
- Uses cosine similarity (we'll explain this below)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

The first time you run this, it downloads the model (~80 MB) and the
tokenizer from HuggingFace. The tokenizer turns text into something the
model can read. After that, both load from a local cache.

## Trying it with simple examples

Let's see how embeddings work on a few examples.

We'll start with a query:

```python
q1 = "Can I still join the course after the start date?"
v1 = model.encode(q1)
```

`v1` is a vector, an array of 384 numbers. Each number stands for some
concept the model learned. We can't read off what any one of them means.
But two vectors with similar values point to texts about similar things.

Encode our document:

```python
d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
dv = model.encode(d)
```

Next, we compare the query against the document using dot product:

```python
v1.dot(dv)
```

We get 0.32.

Now we try an unrelated query:

```python
q2 = "How to install Docker on Windows?"
v2 = model.encode(q2)
```

This time the similarity with the document should be much smaller:

```python
v2.dot(dv)
```

And we get 0.01.

The first score for `q1` vs `d` (0.32) is higher, so that query is more
similar to the document about registration. The second score for `q2`
vs `d` sits near 0, because installing Docker has nothing to do with
registration. A score near 0 means the two vectors are about as
different as they can be.

That's the whole idea behind vector search: similar texts get similar
vectors, and a dot product tells us how similar.

## Cosine similarity

The `all-MiniLM-L6-v2` model outputs normalized vectors - vectors with
unit length. When both vectors are normalized, the dot product equals
cosine similarity. That's why the model documentation says it "uses
cosine similarity."

Cosine similarity measures the angle between two vectors, ignoring
their length:

- 1.0 = same direction (similar)
- 0.0 = perpendicular (unrelated)
- -1.0 = opposite direction (opposite meaning)

Formally, if `theta` is the angle between two vectors, cosine similarity
is `cos(theta)`:

- `cos(0) = 1` - vectors point in the same direction
- `cos(90) = 0` - vectors are perpendicular
- `cos(180) = -1` - vectors point in opposite directions

Because our vectors are normalized, the dot product gives us cosine
similarity directly. This is why we can use `v1.dot(dv)` to compare
texts.

In practice, we rarely get cosine similarity below 0. The embedding
model maps text to a region of the vector space where most vectors
have positive components. There's no concept of "opposite meaning"
that maps to a vector pointing the other way.

[← What is Vector Search](01-intro.md) | [Embedding Our Dataset →](03-embeddings-dataset.md)

```

## Fichier : lessons/09-onnx-embedder.md
```text
# Using ONNX Runtime instead of PyTorch

Video: [Watch this lesson](https://www.youtube.com/watch?v=BMqa4OsCk58&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

When you move to production, you want to cut overhead, both the
dependencies and the size of your deployment. sentence-transformers
drags in PyTorch plus a pile of Nvidia libraries, which is a lot. ONNX
Runtime serves the same model without that weight.

To put a number on it, I created two empty projects. In one I ran `uv
add sentence-transformers`, in the other I set up ONNX Runtime.

Then I measured the virtual environment sizes:

- sentence-transformers: 4.8 GB, 58 packages
- ONNX Runtime: 147 MB, 27 packages

That's 33x smaller for the same embeddings and the same results. Often
we don't even convert the model ourselves. Someone has usually published
an ONNX version we can download.

For development and experiments, sentence-transformers is fine. For
production you want the lighter option.

Let's create a separate project for this lesson:

```bash
mkdir llm-zoomcamp-onnx && cd llm-zoomcamp-onnx
uv init --no-workspace
uv add onnxruntime tokenizers numpy tqdm minsearch
uv add --dev huggingface-hub jupyter
```


`huggingface-hub` is only needed to download the model. At runtime we'll need `onnxruntime`, `tokenizers`, and `numpy`.

Then register a kernel for this project:

```bash
uv run python -m ipykernel install --user --name llm-zoomcamp-onnx --display-name "llm-zoomcamp-onnx"
```


## Downloading the model

We'll use the [download.py](../embed/download.py) script from the
`embed/` directory to fetch the ONNX model from HuggingFace.

Copy it to your project, then run:

```bash
uv run python download.py
```

This creates:

```text
models/
  Xenova/
    all-MiniLM-L6-v2/
      tokenizer.json
      model.onnx
```

You only run this once. After that, the model files are local.

Add the models directory to `.gitignore`:

```text
models/
```

## The Embedder class

We'll use the [embedder.py](../embed/embedder.py) script from the
`embed/` directory for generating embeddings.

Copy it to your project as well.

Under the hood, it does four things:

1. Tokenize - convert text into integer IDs and attention masks
2. Run ONNX model - execute the model graph on CPU
3. Mean pooling - average the token embeddings, weighted by the
   attention mask
4. Normalize - divide by L2 norm so vectors can be compared with
   dot product

You don't need to follow every step inside `embedder.py`. It gives us
the same `encode` interface as before, with none of the PyTorch weight.

## Same pipeline, no PyTorch

Let's repeat the examples from earlier and confirm the numbers match.

First, comparing two queries against a document:

```python
from embedder import Embedder

embed = Embedder()

q1 = "Can I still join the course after the start date?"
q2 = "How to install Docker on Windows?"
d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."

v1 = embed.encode(q1)
v2 = embed.encode(q2)
dv = embed.encode(d)
```

Compute similarities:

```python
v1.dot(dv)
```

And the second similarity:

```python
v2.dot(dv)
```

We get the same result as before. The first score is higher because
the query about joining the course is more similar to the document
about registration.

Next, we embed the FAQ dataset.

If you didn't fetch `ingest.py` earlier, grab it now:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/ingest.py
```

Load the documents:

```python
from ingest import load_faq_data

documents = load_faq_data()
```

Combine question and answer for each document:

```python
texts = [doc["question"] + " " + doc["answer"] for doc in documents]
```

Embed in batches:

```python
from tqdm.auto import tqdm
import numpy as np

batch_size = 50
X = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = embed.encode_batch(batch)
    X.extend(batch_vectors)

X = np.array(X)
```

And search:

```python
query = "Can I still join the course after the start date?"
v_query = embed.encode(query)

scores = X.dot(v_query)
idx = np.argmax(scores)

documents[idx]
```

Same results, same pipeline, but ~33x lighter.

## Available models

All of these work with the same code - just change the model name in
`download.py` and the path in `Embedder()`:

- [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) (384d) - best small general-purpose
- [Xenova/all-MiniLM-L12-v2](https://huggingface.co/Xenova/all-MiniLM-L12-v2) (384d) - better quality, slower
- [Xenova/paraphrase-MiniLM-L6-v2](https://huggingface.co/Xenova/paraphrase-MiniLM-L6-v2) (384d) - paraphrase detection
- [Xenova/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2) (384d) - multilingual
- [Xenova/multilingual-e5-small](https://huggingface.co/Xenova/multilingual-e5-small) (384d) - multilingual retrieval
- [Xenova/multilingual-e5-base](https://huggingface.co/Xenova/multilingual-e5-base) (768d) - stronger multilingual
- [Xenova/bge-small-en-v1.5](https://huggingface.co/Xenova/bge-small-en-v1.5) (384d) - strong retrieval
- [Xenova/bge-base-en-v1.5](https://huggingface.co/Xenova/bge-base-en-v1.5) (768d) - stronger retrieval
- [Xenova/gte-small](https://huggingface.co/Xenova/gte-small) (384d) - lightweight modern model
- [Xenova/gte-base](https://huggingface.co/Xenova/gte-base) (768d) - stronger GTE

To use a different model, add it to `download.py`, run the download,
then update the path:

```python
embed = Embedder("models/Xenova/bge-base-en-v1.5")
vectors = embed.encode("your text here")
print(vectors.shape)
```

Since the runtime only depends on `onnxruntime`, `tokenizers`, and
`numpy`, you can deploy this in minimal environments:

- small Docker images
- serverless functions
- edge devices

[← Vector Search with PGVector](08-pgvector.md) | [Next Steps →](10-next-steps.md)

```

## Fichier : download.py
```text
import os
import shutil
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

ONNX_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/encoder_model.onnx",
    "model.onnx",
]

def download(repo, dest="models"):
    dest = Path(dest) / repo
    dest.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id=repo)
    onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)
    if not onnx_file:
        raise FileNotFoundError(f"No ONNX model found in {repo}")

    for remote, local in [
        ("tokenizer.json", "tokenizer.json"),
        (onnx_file, "model.onnx"),
    ]:
        src = hf_hub_download(repo_id=repo, filename=remote)
        dst = dest / local
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
        else:
            print(f"  exists {dst}")

    onnx_ext = onnx_file + "_data"
    if onnx_ext in files:
        src = hf_hub_download(repo_id=repo, filename=onnx_ext)
        dst = dest / "model.onnx_data"
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
        else:
            print(f"  exists {dst}")

if __name__ == "__main__":
    download("Xenova/all-MiniLM-L6-v2")

```

## Fichier : README.md
```text
# Module 2: Vector Search

In this module, we extend the RAG pipeline from
[module 1](../01-agentic-rag/) with vector search. Vector search
matches documents by semantic meaning instead of exact keyword
overlap. We start from embeddings and end with persistent vector
indexes (sqlitesearch, PGVector) and ONNX-based embedders for
lightweight deployments.

- Code: [code/](code/)
- Embeddings runtime: [embed/](embed/)


## Lessons

The lessons cover vector search end to end, from embeddings to
persistent indexes.

1. [What is Vector Search](lessons/01-intro.md) - Keyword search vs vector search, why it matters
2. [Embeddings](lessons/02-embeddings.md) - Turning text into vectors with sentence-transformers
3. [Embedding Our Dataset](lessons/03-embeddings-dataset.md) - Generating embeddings for the FAQ dataset
4. [Vector Search](lessons/04-vector-search.md) - Vector search with numpy
5. [Vector Search with minsearch](lessons/05-minsearch-vector.md) - In-memory vector search
6. [RAG with Vector Search](lessons/06-rag-vector.md) - Replacing keyword search with vector search in our RAG pipeline
7. [Vector Search with sqlitesearch](lessons/07-sqlitesearch-vector.md) - Persistent vector search backed by SQLite
8. [Vector Search with PGVector](lessons/08-pgvector.md) - Production vector search with PostgreSQL and pgvector
9. [ONNX Embedder](lessons/09-onnx-embedder.md) (Optional) - Using ONNX Runtime instead of PyTorch for embeddings
10. [Next Steps](lessons/10-next-steps.md) - When to use vector search and what's next


## Homework

- [Homework](../cohorts/2026/02-vector-search/homework.md)


## Original workshop recording

This module was taught as a live workshop, which we chopped into the
per-lesson videos above. To watch the full uncut recording:

- [Vector Databases: Embeddings, Semantic Search, and Hybrid Retrieval](https://www.youtube.com/watch?v=BC3NsRUNEIg)


## Old content

Earlier cohorts taught vector search differently. See the archived
materials for the [2024](../cohorts/2024/) and
[2025](../cohorts/2025/) cohorts.


## Notes

- Add your notes above this line

```

## Fichier : embedder.py
```text
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path


class Embedder:
    def __init__(self, path="models/Xenova/all-MiniLM-L6-v2"):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text, normalize=True):
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts, normalize=True):
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled

```

## Fichier : homework_2.md
```text
## Homework: Vector Search

In this homework, we put what we learned in Module 2 into practice.

We'll first turn text into vectors, then search by similarity.
We'll also learn something new and see how to combine vector search with keyword search. We'll skip the RAG part and focus solely on search.

Like in homework 1, our knowledge base is the course lessons themselves.
Each module has a `lessons/` folder of numbered markdown
pages, and we pull them from GitHub. We use the same commit, `8c1834d`,
so everyone works with the exact same 72 pages.

> It's possible your answers won't match exactly. If so, select the closest one.

## Setup

In this homework we won't use the same approach for embedding as in the
module. That is, we won't use the sentence-transformers library. Instead,
we'll use the lightweight embedding approach with the ONNX `Embedder`.

Both approaches produce identical vectors, but the ONNX runtime is far
lighter. It needs no PyTorch and no CUDA, which makes the installation about
30x smaller and lets it run anywhere, including a basic Codespace. We
skimmed through it in the lesson and said we'd cover it in the homework -
so here we are.

We prepare the environment the same way as in the module's
[ONNX Runtime](lessons/09-onnx-embedder.md)
lesson.

Create a fresh project and install the dependencies:

```bash
mkdir llm-zoomcamp-hw2 && cd llm-zoomcamp-hw2
uv init --no-workspace
uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
uv add --dev huggingface-hub jupyter
```

We also need two helper scripts from the `embed/` directory of the course
repo:

- [`download.py`](embed/download.py)
(fetches an ONNX model from HuggingFace) and
- [`embedder.py`](embed/embedder.py) (the `Embedder` class with an `encode` interface)

Let's download them:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/02-vector-search/embed
wget $PREFIX/download.py
wget $PREFIX/embedder.py
```

By default `download.py` fetches `Xenova/all-MiniLM-L6-v2`, the ONNX
version of the `all-MiniLM-L6-v2` model from the lessons:

```bash
uv run python embed/download.py
```

Now we're ready to do the homework.

## Q1. Embedding a query

Embed the following query:

> How does approximate nearest neighbor search work?

The embedder returns a vector of 384 numbers. What's the first value
(`v[0]`)?

* -0.31
* X -0.02
* 0.12
* 0.44

## Loading the data

We pull the lesson pages from the course repository, the same way as in
homework 1. We pin to commit `8c1834d` so everyone works with the same
data.

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
```

Each document is a dictionary with a `filename` and `content`, and there
are 72 pages.

## Q2. Cosine similarity

The embedder returns normalized vectors, so the dot product between two
of them is their cosine similarity.

Take the page `02-vector-search/lessons/07-sqlitesearch-vector.md`, embed
its `content`, and compute the cosine similarity with the query vector
from Q1. What do you get?

* 0.07
* X 0.37
* 0.68
* 0.92

## Q3. Chunking and search by hand

A full page covers several topics, which waters down its embedding.

We chunk the pages the same way as in homework 1:

```python
from gitsource import chunk_documents
chunks = chunk_documents(documents, size=2000, step=1000)
```

We embed every chunk's `content` with `encode_batch`, stack the vectors
into a matrix `X`, and score the Q1 query against all chunks:

```python
scores = X.dot(v)
```

Which file does the highest-scoring chunk belong to (its `filename`)?

* `02-vector-search/lessons/03-embeddings-dataset.md`
* `02-vector-search/lessons/06-rag-vector.md`
* X `02-vector-search/lessons/07-sqlitesearch-vector.md`
* `02-vector-search/lessons/09-onnx-embedder.md`

## Q4. Vector search with minsearch

We've done vector search by hand, which is good for learning, but it's not
what we do in practice. In practice we use libraries.

Let's use `VectorSearch` from minsearch and run a search for the following
query:

> What metric do we use to evaluate a search engine?

Which file is the `filename` of the first result?

* `02-vector-search/lessons/04-vector-search.md`
* X `04-evaluation/lessons/05-search-metrics.md`
* `04-evaluation/lessons/13-llm-as-judge.md`
* `05-monitoring/lessons/04-metrics.md`

## Q5. Text search vs vector search

Vector search matches by meaning, keyword search by exact words.

Let's compare them. Index
the same chunks with `Index` from minsearch. Use `content` as a
text field.

Run both searches for this query:

> How do I store vectors in PostgreSQL?

Take the top 5 results from each method. Which file shows up in the
vector results but not in the text results?

* `02-vector-search/lessons/01-intro.md`
* `02-vector-search/lessons/02-embeddings.md`
* X `02-vector-search/lessons/08-pgvector.md`
* `03-orchestration/lessons/05-rag.md`

## Q6. Hybrid search

Both vector and text search have their strengths and weaknesses. Vector
search matches by meaning, so it finds relevant pages even when they use
words different from the query. But it can miss exact terms like names,
codes, or rare keywords. Text search is the opposite: it nails exact words
but misses paraphrases and synonyms.

We don't have to pick one or the other - we can use both and merge their
results. This approach is called "hybrid search".

Each search produces its own ranked list, so we need a way to combine them
into one. In this homework we use Reciprocal Rank Fusion (RRF). It ignores
the raw scores from each method, which live on different scales and aren't
directly comparable. Instead, it looks only at the position of each
document in each list.

Every document scores by its position (`rank`, starting at 0) in each
list, and we sum the scores across lists with a constant `k = 60`:

```text
RRF(d) = sum over lists of  1 / (k + rank(d))
```

"Sum over lists" means we go through every ranked list and, for each list
where the document appears, add its `1 / (k + rank)` contribution. A
document found by both searches collects a score from each list, while one
found by only a single search collects just one.

The constant `k` controls how much the exact rank matters. A larger `k`
flattens the gap between positions, so the difference between rank 0 and
rank 5 counts for less. A smaller `k` does the opposite: it sharpens that
gap, so being at the top of a list matters much more.

The value 60 comes from the original RRF paper and is the usual default.
You rarely need to tune it. Lower it when only the top results matter.
Raise it to reward documents that appear across many lists, even when they
never quite reach the top.

A document that ranks well in both lists ends up higher than one that's
only strong in a single list.

```python
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]
```

Now run the query `"How do I give the model access to tools?"`
with vector and text search and fuse the results with `rrf`:

```python
results = rrf([vector_results, text_results])
```

Which file is ranked first after RRF?

* `01-agentic-rag/lessons/01-intro.md`
* X `01-agentic-rag/lessons/13-function-calling.md`
* `01-agentic-rag/lessons/14-agentic-loop.md`
* `01-agentic-rag/lessons/16-other-frameworks.md`

Notice that this file isn't first in either search on its own - it wins
because it ranks high in both.

## Selecting the best approach

By now you can search several ways:

- vector search
- keyword search
- hybrid search

Which is the best way? 

The right choice depends on your data, and the way to decide is to
measure.

We cover how to evaluate and compare search approaches in the
[evaluation module](../../../04-evaluation/lessons/04-search-evaluation.md),
and you'll do exactly that in the [evaluation homework](../04-evaluation/homework.md).

## Learning in Public

We encourage everyone to share what they learned. This is called "learning in public".

Read more about the benefits [here](https://alexeyondata.substack.com/p/benefits-of-learning-in-public-and) and in the [course's learning in public guide](https://datatalks.club/docs/courses/zoomcamp-logistics/learning-in-public/).

### Example post for LinkedIn

Tag [@Alexey Grigorev](https://www.linkedin.com/in/agrigorev/) and [@DataTalksClub](https://www.linkedin.com/company/datatalks-club/) in your post - we'll like and comment to give your post more reach.

```
🚀 Module 2 of LLM Zoomcamp by @DataTalksClub complete!

Just finished Module 2 - Vector Search. Learned how to:

✅ Turn text into embeddings with a lightweight ONNX model
✅ Build vector search from scratch with numpy
✅ Use minsearch for vector search, with chunking for long pages
✅ Compare keyword and vector search, and combine them with hybrid search (RRF)

Here's my homework solution: <LINK>

Following along with this amazing free course by @Alexey Grigorev - who else is learning to build with LLMs?

You can sign up here: https://github.com/DataTalksClub/llm-zoomcamp/
```

### Example post for X

```
🔍 Module 2 of LLM Zoomcamp done!

- Embeddings with a lightweight ONNX model
- Vector search from scratch with numpy
- minsearch + chunking
- Keyword vs vector, and hybrid search with RRF

My solution: <LINK>

Free course by @Al_Grigor & @DataTalksClub: https://github.com/DataTalksClub/llm-zoomcamp/
```

## Submit the results

* Submit your results here: https://courses.datatalks.club/llm-zoomcamp-2026/homework/hw2
* It's possible your answers won't match exactly. If so, select the closest one.

```

