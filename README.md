# RAG Learning

This repository contains examples and Jupyter notebooks for learning Retrieval-Augmented Generation (RAG) using LlamaIndex.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/donniv86/rag-learning.git
   cd rag-learning
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key in a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Jupyter Notebooks for Hands-On Learning

The repository includes the following Jupyter notebooks for interactive learning:

- **indexing_basics.ipynb**: Learn basic indexing techniques with LlamaIndex.
- **multi_modal_examples.ipynb**: Explore indexing and querying text, PDF, and image files.
- **embedding_techniques.ipynb**: Understand different embedding models and their characteristics.

### How to Use the Notebooks

1. Start JupyterLab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. Open the desired notebook and run each cell interactively to see the results and experiment with the code.

## Additional Resources

- [LlamaIndex Documentation](https://docs.llama-index.ai/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.