FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e ".[rag,ui,serving]" torch --index-url https://download.pytorch.org/whl/cpu || pip install --no-cache-dir -e ".[rag,ui,serving]"

EXPOSE 7860
CMD ["python", "-m", "blockrank_rag.cli", "--help"]
# For gradio demo: docker run ... python examples/simple_rag_demo.py  (or add app.py)
