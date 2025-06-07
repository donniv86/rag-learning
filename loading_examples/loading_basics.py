from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import SimpleDirectoryReader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_sample_text_file():
    """Create a sample text file for demonstration"""
    os.makedirs("data", exist_ok=True)
    with open("data/sample.txt", "w") as f:
        f.write("""
        LlamaIndex is a powerful framework for building RAG applications.
        It provides various tools for loading, processing, and querying data.
        The loading stage is the first step in building a RAG application.
        You can load data from various sources like text files, PDFs, and databases.
        """)

def example_1_basic_document_creation():
    """Example 1: Creating documents from text directly"""
    print("\nExample 1: Creating documents from text directly")

    # Create documents from text
    text_list = [
        "LlamaIndex is a powerful framework for RAG applications.",
        "It helps in building AI applications with your own data.",
        "The framework provides various tools for data processing."
    ]

    documents = [Document(text=t) for t in text_list]

    # Create an index
    index = VectorStoreIndex.from_documents(documents)

    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query("What is LlamaIndex?")
    print(f"Query response: {response}")

def example_2_loading_from_file():
    """Example 2: Loading documents from a file"""
    print("\nExample 2: Loading documents from a file")

    # Create a sample file
    create_sample_text_file()

    # Load documents from the file
    documents = SimpleDirectoryReader("data").load_data()

    # Create an index
    index = VectorStoreIndex.from_documents(documents)

    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query("What can you do with LlamaIndex?")
    print(f"Query response: {response}")

def example_3_node_parsing():
    """Example 3: Parsing documents into nodes"""
    print("\nExample 3: Parsing documents into nodes")

    # Create a document
    text = """
    LlamaIndex provides various tools for data processing.
    You can split documents into smaller chunks called nodes.
    Nodes are useful for more granular processing and retrieval.
    The SentenceSplitter helps in creating nodes from documents.
    """
    document = Document(text=text)

    # Parse the document into nodes
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents([document])

    # Print node information
    print(f"Number of nodes created: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"\nNode {i+1}:")
        print(f"Text: {node.text}")
        print(f"Metadata: {node.metadata}")

def main():
    print("Starting LlamaIndex Loading Examples...")

    # Run examples
    example_1_basic_document_creation()
    example_2_loading_from_file()
    example_3_node_parsing()

if __name__ == "__main__":
    main()