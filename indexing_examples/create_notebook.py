import json
import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = [
    nbf.v4.new_code_cell("""# Install required packages (run this cell if you see import errors)
!pip install numpy scikit-learn
"""),
    nbf.v4.new_markdown_cell("# RAG Evaluation and Metrics\n\nThis notebook explores various techniques for evaluating RAG systems."),
    nbf.v4.new_code_cell("# Import required libraries\nimport numpy as np\nfrom sklearn.metrics import precision_score, recall_score"),
    nbf.v4.new_markdown_cell("## Exercise 1: Retrieval Metrics\n\nEvaluate the quality of document retrieval in RAG systems."),
    nbf.v4.new_code_cell("""class RetrievalEvaluator:
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth

    def calculate_precision(self, retrieved_docs, k=5):
        relevant_docs = set(self.ground_truth)
        retrieved_docs = set(retrieved_docs[:k])

        if not retrieved_docs:
            return 0.0

        return len(relevant_docs.intersection(retrieved_docs)) / len(retrieved_docs)

    def calculate_recall(self, retrieved_docs, k=5):
        relevant_docs = set(self.ground_truth)
        retrieved_docs = set(retrieved_docs[:k])

        if not relevant_docs:
            return 0.0

        return len(relevant_docs.intersection(retrieved_docs)) / len(relevant_docs)

# Test retrieval metrics
ground_truth = ['doc1', 'doc2', 'doc3']
retrieved_docs = ['doc1', 'doc4', 'doc2', 'doc5', 'doc3']

evaluator = RetrievalEvaluator(ground_truth)

print(f"Precision@5: {evaluator.calculate_precision(retrieved_docs):.3f}")
print(f"Recall@5: {evaluator.calculate_recall(retrieved_docs):.3f}")""")
]

nb['cells'] = cells

# Write the notebook to a file
with open('rag_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)