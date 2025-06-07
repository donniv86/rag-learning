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

## Loading Examples

The `loading_examples` directory contains examples demonstrating different ways to load data in LlamaIndex:

1. `loading_basics.py`: Shows three different ways to load data:
   - Creating documents from text directly
   - Loading documents from files
   - Parsing documents into nodes

To run the examples:
```bash
cd loading_examples
python loading_basics.py
```

## What You'll Learn

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