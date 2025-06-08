from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_sample_data():
    """Create sample data files for demonstration"""
    os.makedirs("data", exist_ok=True)

    # Create a sample text file about RAG
    with open("data/rag_intro.txt", "w") as f:
        f.write("""
        Retrieval-Augmented Generation (RAG) is a powerful technique in AI.
        It combines the strengths of retrieval-based and generation-based approaches.
        RAG helps in creating more accurate and contextually relevant responses.
        The process involves retrieving relevant information and using it to generate responses.
        """)

    # Create a sample text file about indexing
    with open("data/indexing_info.txt", "w") as f:
        f.write("""
        Indexing is a crucial step in the RAG pipeline.
        It involves creating a searchable structure from your data.
        Vector embeddings are used to represent text in a numerical format.
        These embeddings help in finding semantically similar content.
        """)

def example_1_basic_indexing():
    """Example 1: Basic indexing with default settings"""
    print("\nExample 1: Basic Indexing")

    # Load documents
    documents = SimpleDirectoryReader("data").load_data()

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query("What is RAG?")
    print(f"Query response: {response}")

def example_2_custom_node_parsing():
    """Example 2: Custom node parsing with metadata"""
    print("\nExample 2: Custom Node Parsing")

    # Load documents
    documents = SimpleDirectoryReader("data").load_data()

    # Create custom node parser
    parser = SentenceSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separator="\n"
    )

    # Parse documents into nodes
    nodes = parser.get_nodes_from_documents(documents)

    # Print node information
    print(f"Number of nodes created: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"\nNode {i+1}:")
        print(f"Text: {node.text}")
        print(f"Metadata: {node.metadata}")

def example_3_metadata_indexing():
    """Example 3: Indexing with custom metadata"""
    print("\nExample 3: Indexing with Custom Metadata")

    # Create documents with custom metadata
    documents = [
        Document(
            text="RAG combines retrieval and generation for better AI responses.",
            metadata={"source": "rag_intro.txt", "category": "introduction"}
        ),
        Document(
            text="Vector embeddings help in finding similar content.",
            metadata={"source": "indexing_info.txt", "category": "technical"}
        )
    ]

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Query with metadata filtering
    query_engine = index.as_query_engine(
        similarity_top_k=2,
        response_mode="compact"
    )

    # Query the index
    response = query_engine.query("How does RAG work?")
    print(f"Query response: {response}")

    # Print metadata mode
    print("\nMetadata Mode Example:")
    print(f"Full metadata: {documents[0].get_content(metadata_mode=MetadataMode.ALL)}")
    print(f"No metadata: {documents[0].get_content(metadata_mode=MetadataMode.NONE)}")

def main():
    print("Starting LlamaIndex Indexing Examples...")

    # Create sample data
    create_sample_data()

    # Run examples
    example_1_basic_indexing()
    example_2_custom_node_parsing()
    example_3_metadata_indexing()

if __name__ == "__main__":
    main()