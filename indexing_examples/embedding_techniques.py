from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
import numpy as np
from typing import List
import time

# Load environment variables
load_dotenv()

def create_sample_documents() -> List[Document]:
    """Create sample documents for testing different embedding techniques"""
    return [
        Document(text="RAG combines retrieval-based and generation-based approaches for better AI responses."),
        Document(text="Vector databases store embeddings for efficient similarity search."),
        Document(text="Embeddings convert text into numerical vectors that capture semantic meaning."),
        Document(text="Semantic search finds relevant content based on meaning rather than keywords."),
        Document(text="Chunking breaks down large documents into smaller, manageable pieces.")
    ]

def compare_embeddings(text: str, embedding_model1, embedding_model2):
    """Compare embeddings from two different models"""
    # Get embeddings
    embedding1 = embedding_model1.get_text_embedding(text)
    embedding2 = embedding_model2.get_text_embedding(text)

    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return similarity

def example_1_basic_embeddings():
    """Example of basic embedding creation and comparison"""
    print("\n=== Basic Embedding Example ===")

    # Initialize embedding models
    openai_embedding = OpenAIEmbedding()
    huggingface_embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Sample text
    text = "RAG systems use embeddings for semantic search"

    # Get embeddings
    openai_embedding_vector = openai_embedding.get_text_embedding(text)
    huggingface_embedding_vector = huggingface_embedding.get_text_embedding(text)

    print(f"OpenAI Embedding dimension: {len(openai_embedding_vector)}")
    print(f"HuggingFace Embedding dimension: {len(huggingface_embedding_vector)}")

    # Compare embeddings
    similarity = compare_embeddings(text, openai_embedding, huggingface_embedding)
    print(f"Similarity between OpenAI and HuggingFace embeddings: {similarity:.4f}")

def example_2_chunking_and_embeddings():
    """Example of how chunking affects embeddings"""
    print("\n=== Chunking and Embeddings Example ===")

    # Initialize embedding model
    embedding_model = OpenAIEmbedding()

    # Create a longer text
    long_text = """
    RAG (Retrieval-Augmented Generation) is a powerful technique that combines the strengths of
    retrieval-based and generation-based approaches. It first retrieves relevant information from
    a knowledge base and then uses that information to generate more accurate and contextually
    relevant responses. This approach helps reduce hallucinations and improves the reliability
    of AI-generated content.
    """

    # Create different chunking strategies
    chunker1 = SentenceSplitter(chunk_size=50, chunk_overlap=0)
    chunker2 = SentenceSplitter(chunk_size=100, chunk_overlap=20)

    # Split text into chunks
    chunks1 = chunker1.split_text(long_text)
    chunks2 = chunker2.split_text(long_text)

    print(f"Number of chunks (strategy 1): {len(chunks1)}")
    print(f"Number of chunks (strategy 2): {len(chunks2)}")

    # Get embeddings for first chunk using both strategies
    embedding1 = embedding_model.get_text_embedding(chunks1[0])
    embedding2 = embedding_model.get_text_embedding(chunks2[0])

    # Compare embeddings
    similarity = np.dot(np.array(embedding1), np.array(embedding2)) / (
        np.linalg.norm(np.array(embedding1)) * np.linalg.norm(np.array(embedding2))
    )
    print(f"Similarity between different chunking strategies: {similarity:.4f}")

def example_3_semantic_search():
    """Example of semantic search using embeddings"""
    print("\n=== Semantic Search Example ===")

    # Create sample documents
    documents = create_sample_documents()

    # Create index with OpenAI embeddings
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine()

    # Test queries
    queries = [
        "How does RAG work?",
        "What are vector databases used for?",
        "How do we handle large documents?"
    ]

    for query in queries:
        start_time = time.time()
        response = query_engine.query(query)
        end_time = time.time()

        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        print(f"Time taken: {(end_time - start_time):.2f} seconds")

def example_4_embedding_visualization():
    """Example of visualizing embedding relationships"""
    print("\n=== Embedding Visualization Example ===")

    # Create sample documents
    documents = create_sample_documents()

    # Get embeddings for all documents
    embedding_model = OpenAIEmbedding()
    embeddings = [embedding_model.get_text_embedding(doc.text) for doc in documents]

    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(documents), len(documents)))
    for i in range(len(documents)):
        for j in range(len(documents)):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[j])
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarity_matrix[i][j] = similarity

    # Print similarity matrix
    print("\nDocument Similarity Matrix:")
    print("Document 1: RAG combines retrieval-based and generation-based approaches")
    print("Document 2: Vector databases store embeddings for efficient similarity search")
    print("Document 3: Embeddings convert text into numerical vectors")
    print("Document 4: Semantic search finds relevant content based on meaning")
    print("Document 5: Chunking breaks down large documents")
    print("\nSimilarity scores (higher means more similar):")
    print(similarity_matrix)

def main():
    # Set OpenAI API key
    Settings.embed_model = OpenAIEmbedding()

    # Run examples
    example_1_basic_embeddings()
    example_2_chunking_and_embeddings()
    example_3_semantic_search()
    example_4_embedding_visualization()

if __name__ == "__main__":
    main()