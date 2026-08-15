# Neural Memory LLM

A graph-based associative memory layer for LLM applications.

Neural Memory LLM ingests documents, converts them into interconnected concepts and relationships, and recalls relevant information through **spreading activation** before asking an LLM to generate the final answer.

Instead of treating every document chunk as an isolated vector result, the system builds a persistent memory graph that can connect information across documents.

## What is Neural Memory LLM?

The project combines three layers:

```text
Documents
   ↓
Entity + Relationship Extraction
   ↓
Persistent Memory Graph
   ↓
Spreading Activation
   ↓
Relevant Memories + Source Evidence
   ↓
LLM
   ↓
Natural-Language Answer
```

The **memory graph** acts as the persistent knowledge layer.

The **LLM** acts as the language and reasoning layer.

This separation allows the knowledge base to grow without retraining the underlying LLM every time a new document is added.

## Why is it different?

Traditional RAG commonly works like:

```text
Document → Chunks → Embeddings → Vector Search → Top-K Chunks → LLM
```

Neural Memory LLM adds an associative memory layer:

```text
Document → Concepts → Relationships → Memory Graph
                                      ↓
                              Spreading Activation
                                      ↓
                                Memory Recall
                                      ↓
                                 LLM Answer
```

### Advantages

**Associative recall**

Related concepts can activate connected memories, allowing the system to follow relationships across multiple documents.

**Multi-hop reasoning**

A question does not have to match the exact paragraph containing the answer. The system can follow connected concepts across several relationships.

**Persistent memory**

The graph is stored locally, so the system remembers previously ingested knowledge after restarting the application.

**Cross-document connections**

Information from separate documents can become part of the same memory network.

**Inspectable memory**

The graph can be visualized and inspected instead of keeping retrieval completely hidden.

**Provider-independent memory**

The memory layer is separate from the LLM provider. OpenRouter, Ollama, or another provider can be used without rebuilding the memory graph.

## Memory Graph

The following graph shows the kind of interconnected memory structure produced from the ingested documents.

![Neural Memory Graph](data/neural-memeory-llm.jpg)

The graph contains concepts such as:

* Federal Reform Coalition
* Civic Services Reform Act
* National Development Party
* Sofia Bennett
* Riverland Regional Council
* Parliamentary Review

The important part is not only the individual nodes, but the relationships connecting them.

A query can activate one concept and spread activation through related concepts to recall information stored elsewhere in the graph.

## Demo

A full application walkthrough will be added here.

> **Demo video:** Add your recorded demo as `assets/demo.mp4`.

Once the video is added, you can use:

```html
<video src="assets/demo.mp4" controls width="900"></video>
```

## Local Setup

### 1. Clone the project

```bash
git clone <your-repository-url>
cd graph-memory
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure OpenRouter

Create `.env` from the example file:

```powershell
Copy-Item .env.example .env
```

Set:

```env
LLM_PROVIDER=openrouter

OPENROUTER_API_KEY=YOUR_API_KEY
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openrouter/free
```

You can replace `openrouter/free` with another model available to your OpenRouter account.

### 5. Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## How to use it

### Ingest documents

Upload supported files such as:

```text
PDF
DOCX
TXT
Markdown
```

Click:

```text
Build / Update Memory
```

The pipeline performs:

```text
Document loading
    ↓
Chunking
    ↓
Entity extraction
    ↓
Relationship extraction
    ↓
Memory graph update
    ↓
Persistent storage
```

### Ask questions

After the memory is built, open the Chat tab and ask questions about the uploaded knowledge.

For example:

```text
How are the Federal Reform Coalition and the 2024 parliamentary review connected?
```

The system recalls connected memories first and then sends the relevant evidence to the LLM.

## Project Structure

```text
graph-memory/
├── app.py
├── config.py
├── pipeline.py
├── extraction/
├── graph/
├── retrieval/
├── generation/
├── ingestion/
├── storage/
├── visualization/
├── ui/
├── data/
└── tests/
```

The most important modules are:

```text
graph/
    Persistent memory graph

retrieval/
    Spreading activation and associative recall

generation/
    LLM answer generation

extraction/
    Entity and relationship extraction
```

## Core Idea

The project is built around a simple principle:

> **The LLM should not have to remember everything. The memory layer should decide what is worth recalling.**

The memory graph stores the relationships.

Spreading activation determines what becomes relevant.

The LLM turns the recalled information into a natural-language response.

```text
Activate
   ↓
Associate
   ↓
Recall
   ↓
Reason
   ↓
Answer
```

## Future Work : Suggest me Improvement at ![Email](https://img.shields.io/badge/Email-Contact%20Me-blue?logo=gmail)](mailto:kaleayush8055@gmail.com)

Planned improvements include:

* Better entity resolution
* Semantic query activation
* Temporal memory
* Contradiction handling
* Memory consolidation
* More advanced synaptic learning
* Graph database support
* Vector + graph hybrid retrieval
* Large-scale background ingestion


