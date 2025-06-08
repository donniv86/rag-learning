from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.readers.file import PDFReader, ImageReader
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image

# Load environment variables
load_dotenv()

def example_1_text_indexing():
    """Example of indexing text files"""
    print("\n=== Text Indexing Example ===")
    print("Focus: Direct text processing with semantic understanding")

    # Create sample text document
    text = """
    RAG (Retrieval-Augmented Generation) combines the power of retrieval-based and generation-based approaches.
    Key Benefits:
    1. Improved accuracy and reliability
    2. Access to up-to-date information
    3. Ability to cite sources
    4. Reduced hallucinations
    5. Domain-specific knowledge integration
    """
    document = Document(text=text)

    # Create index
    index = VectorStoreIndex.from_documents([document])

    # Create query engine
    query_engine = index.as_query_engine()

    # Test queries to show semantic understanding
    queries = [
        "What are the main advantages of RAG?",
        "How does RAG improve reliability?",
        "What is the relationship between RAG and hallucinations?"
    ]

    for query in queries:
        response = query_engine.query(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

def example_2_pdf_indexing():
    """Example of indexing PDF files"""
    print("\n=== PDF Indexing Example ===")
    print("Focus: Structured document processing with metadata")

    # Initialize PDF reader
    reader = PDFReader()

    # Load PDF documents
    documents = reader.load_data("data/sample.pdf")

    # Print document metadata
    print("\nDocument Metadata:")
    for doc in documents:
        print(f"Page Number: {doc.metadata.get('page_label', 'N/A')}")
        print(f"File Name: {doc.metadata.get('file_name', 'N/A')}")
        print(f"File Type: {doc.metadata.get('file_type', 'N/A')}")
        print("---")

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine()

    # Test queries to show structured understanding
    queries = [
        "What is the main topic of the PDF?",
        "What are the key sections in the document?",
        "What formatting is used in the document?"
    ]

    for query in queries:
        response = query_engine.query(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

def example_3_image_indexing():
    """Example of indexing image files"""
    print("\n=== Image Indexing Example ===")
    print("Focus: OCR text extraction and visual content understanding")

    # Use OCR to extract text from the image
    image = Image.open("data/test.png")
    ocr_text = pytesseract.image_to_string(image)

    # Print raw OCR output
    print("\nRaw OCR Output:")
    print(ocr_text)

    # Print OCR confidence (if available)
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidence = [int(x) for x in ocr_data['conf'] if x != '-1']
        if confidence:
            print(f"\nOCR Confidence: {sum(confidence)/len(confidence):.2f}%")
    except:
        print("\nOCR confidence data not available")

    # Create a document from the OCR text
    document = Document(text=ocr_text)

    # Create index from the OCR text document
    index = VectorStoreIndex.from_documents([document])

    # Create query engine
    query_engine = index.as_query_engine()

    # Test queries to show visual content understanding
    queries = [
        "What is the main topic of this image?",
        "What are the key details in the image?",
        "What is the call to action in the image?"
    ]

    for query in queries:
        response = query_engine.query(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

def main():
    # Set OpenAI API key
    Settings.embed_model = OpenAIEmbedding()

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Run examples
    example_1_text_indexing()
    example_2_pdf_indexing()
    example_3_image_indexing()

if __name__ == "__main__":
    main()