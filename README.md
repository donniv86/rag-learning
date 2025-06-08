# RAG Learning Project

This project demonstrates various concepts of RAG (Retrieval-Augmented Generation) using LlamaIndex.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Examples

### 1. Loading Examples
The `loading_examples` directory contains examples demonstrating different ways to load data in LlamaIndex:

1. `loading_basics.py`: Shows three different ways to load data:
   - Creating documents from text directly
   - Loading documents from files
   - Parsing documents into nodes

### 2. Indexing Examples
The `indexing_examples` directory contains examples demonstrating different indexing strategies:

1. `indexing_basics.py`: Shows three different indexing approaches:
   - Basic indexing with default settings
   - Custom node parsing with metadata
   - Indexing with custom metadata and filtering

## What You'll Learn

### Loading Stage
1. **Basic Document Creation**
   - How to create Document objects from text
   - How to build a simple index
   - How to query the index

2. **Loading from Files**
   - How to load data from text files
   - Using SimpleDirectoryReader
   - Creating and querying an index from file data

3. **Node Parsing**
   - How to split documents into nodes
   - Understanding node structure
   - Working with node metadata

### Indexing Stage
1. **Basic Indexing**
   - Creating vector indices from documents
   - Default indexing behavior
   - Simple querying

2. **Custom Node Parsing**
   - Configuring chunk size and overlap
   - Custom separators
   - Node metadata management

3. **Metadata Indexing**
   - Adding custom metadata to documents
   - Using metadata for filtering
   - Different metadata modes

## Running the Examples

To run the loading examples:
```bash
cd loading_examples
python loading_basics.py
```

To run the indexing examples:
```bash
cd indexing_examples
python indexing_basics.py
```