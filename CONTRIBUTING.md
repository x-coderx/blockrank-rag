# Contributing to blockrank-rag

Thank you for your interest in improving this project! `blockrank-rag` is a practical, production-oriented toolkit built on top of the BlockRank research ideas.

## Getting Started

1. Clone the repository (including the reference implementation):
   ```bash
   git clone https://github.com/yourusername/blockrank-rag.git
   cd blockrank-rag
   ```

2. Install with development dependencies:
   ```bash
   pip install -e ".[dev,all]"
   ```

3. (Optional) Set up the original research code for reference:
   ```bash
   git clone https://github.com/nilesh2797/BlockRank.git blockrank-original
   cd blockrank-original && pip install -e .
   ```

## How to Contribute

- **Bug reports & feature requests**: Open an issue with clear reproduction steps or a detailed proposal.
- **Code contributions**: 
  1. Fork the repo and create a feature branch.
  2. Make your changes (add tests where appropriate).
  3. Run linting and tests:
     ```bash
     ruff check .
     pytest -q
     ```
  4. Submit a pull request.

### Areas of Interest
- Support for additional LLMs (Llama-3, Qwen2, Gemma, etc.)
- Improved chunking strategies and aggregation methods
- More comprehensive BEIR / custom evaluation scripts
- Better integration with popular RAG frameworks (LangChain, LlamaIndex, etc.)
- Training pipeline improvements (full aux loss, LoRA best practices)

When modifying collation, attention scoring, or block tokenization logic, please review the reference implementation in `blockrank-original/` to maintain compatibility with the original design.

## Code Style
- Follow PEP 8.
- Run `ruff check .` before committing.
- Add type hints where reasonable.
- Keep new public APIs documented.

## Questions?
Feel free to open a discussion or issue. We're happy to help you get started.

---

This project follows the [MIT License](LICENSE).