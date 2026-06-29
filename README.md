# LLM Zoomcamp 2026 - Module 2: Vector Search 🔍

My solution repository for **Homework 2** of the Vector Search module in the 2026 edition of the LLM Zoomcamp (organized by DataTalksClub).

This module explores the core concepts of semantic (vector) search, lightweight production-ready deployment environments (ONNX Runtime), and search engine merging strategies (Hybrid Search via RRF).

---

## 🚀 Project Overview

The goal of this homework is to manipulate vector representations (embeddings) of text documents extracted from the course lessons (pinned to commit `8c1834d`) to efficiently retrieve answers to user queries without relying solely on exact keyword matching.

### Tech Stack & Tools Used:
* **Package Management:** `uv` (modern, blazing-fast dependency manager)
* **Lightweight Embeddings:** `onnxruntime` & `tokenizers` (Model: `Xenova/all-MiniLM-L6-v2`)
* **Search Engine:** `minsearch` (In-memory vector & text search tool)
* **Data Processing:** `numpy`, `tqdm`, `gitsource`

---

## 📂 Directory Structure

* `homework_2.ipynb`: Jupyter Notebook containing all the answers, implementation code, and calculations for the assignment.
* `embedder.py`: Utility class using ONNX Runtime to generate text embeddings 33x lighter than standard PyTorch environments.
* `download.py`: Automation script to fetch the required ONNX model directly from Hugging Face.

---

## 🧠 Key Concepts Implemented

### 1. PyTorch-Free Embeddings (ONNX Runtime)
Leveraged the compact 384-dimensional `all-MiniLM-L6-v2` model. Transitioning to ONNX Runtime shrinks the virtual environment size from **4.8 GB** (with PyTorch/CUDA) down to just **147 MB** while producing mathematically identical vectors and similarity scores. This approach is ideal for serverless or edge production deployments.

### 2. Vector Search "From Scratch"
Calculated cosine similarity via NumPy matrix multiplication (`X.dot(v)`). This highlights the performance benefits of NumPy's optimized C-backend over native Python loops when executing Nearest Neighbor (NN) search queries.

### 3. Document Chunking
Divided long markdown lesson pages into smaller text chunks of 2000 characters with a 1000-character overlapping step. This prevents semantic dilution, ensuring the embedding captures specific, granular topics effectively.

### 4. Hybrid Search & RRF (Reciprocal Rank Fusion)
Implemented the Reciprocal Rank Fusion algorithm to combine the best of both worlds: the surgical precision of exact keyword matching and the contextual flexibility of semantic vector search. Scores from both search methods are fused seamlessly using a constant rank penalty of $k = 60$.

---

## 🛠️ Installation & Setup

Ensure you have `uv` installed on your local system.

1. Synchronize the dependencies and initialize the virtual environment:
   ```bash
   uv sync

```

2. Download the ONNX model files locally:
```bash
uv run python download.py

```


3. Spin up Jupyter to run the notebook:
```bash
uv run jupyter notebook

```



---

🎓 *Course provided by [Alexey Grigorev](https://www.linkedin.com/in/agrigorev/) / [DataTalksClub*](https://datatalks.club/)

```